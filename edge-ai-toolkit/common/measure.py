"""
common/measure.py -- measurement plumbing shared by every benchmark here.

These are the details that decide whether a benchmark number means anything.
They live in one place because getting them wrong in two places
independently is how two directories end up quietly disagreeing.

Each function documents the trap it exists to avoid, because none of them is
obvious and I got two of them wrong the first time.
"""

from __future__ import annotations

import os
import platform
import re
import resource
import subprocess
import sys
from pathlib import Path


def peak_rss_mib():
    """
    Peak resident set size of THIS process, in MiB.

    Reads VmHWM from /proc/self/status rather than
    getrusage(RUSAGE_SELF).ru_maxrss.

    This is a correctness fix, not a style preference. Linux keeps ru_maxrss
    in the signal struct, and that struct SURVIVES execve. A child launched
    from a parent that has touched 2 GiB reports 2 GiB as its own peak
    before running a single line of its own code. Measured directly:

        parent peak 1212 MiB -> child ru_maxrss 1212 MiB, child VmHWM 11 MiB

    VmHWM lives in the mm_struct, which execve replaces, so it starts clean.
    Any harness that measures work in subprocesses -- as these do, precisely
    so that peaks do not contaminate each other -- must use VmHWM or it will
    report the parent's memory for every case after the first.

    Falls back to getrusage where /proc is unavailable (macOS, BSD).
    """
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024          # kB -> MiB
    except OSError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / 1024**2) if sys.platform == "darwin" else (r / 1024)


def drop_cache(path):
    """
    Evict a file from the page cache. No root required.

    Without this, the first read populates the cache and every subsequent
    read is served from RAM, so you measure memory bandwidth and conclude
    that all formats and all engines are equally fast.

    Recurses into directories, which matters for partitioned Parquet
    datasets where the "file" is a tree.
    """
    p = Path(path)
    files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
    ok = True
    for f in files:
        try:
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except (OSError, AttributeError):
            ok = False
    return ok


def fsync_path(path):
    """
    Force a written file to the device, and its directory entry with it.

    Timing a write only as far as the page cache is not timing a write. It
    made pickle appear to write 88 MB in 0.043 s -- over 2 GB/s, faster than
    any SD card physically accepts -- and it made zstd look 81% more
    expensive than it actually is.

    Worse, an unflushed write does not only report its own time wrongly. The
    kernel flushes those pages later, competing with whatever you measure
    next, so it corrupts the following read as well.
    """
    p = Path(path)
    targets = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
    for f in targets:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    dfd = os.open(str(p if p.is_dir() else p.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


# =========================================================================
# environment reporting
# =========================================================================
# A benchmark that does not state what it ran on is not reproducible. An SD
# card and an NVMe SSD differ by more than an order of magnitude on write
# throughput, and a thermally throttled Pi produces slow numbers that have
# nothing to do with what is being compared.

def _read(path, default=""):
    try:
        return Path(path).read_text(errors="replace").strip("\x00 \n")
    except OSError:
        return default


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def root_storage(where=None):
    """What is the working directory sitting on? SD, USB, NVMe?"""
    try:
        dev = _cmd(["findmnt", "-no", "SOURCE", "--target",
                    str(where or Path.cwd())])
        base = re.sub(r"p?\d+$", "", Path(dev).name) if dev else ""
        if not base:
            return dev or "unknown"
        model = _read(f"/sys/block/{base}/device/model") or \
                _read(f"/sys/block/{base}/device/name")
        rot = _read(f"/sys/block/{base}/queue/rotational")
        kind = ("SD/eMMC" if base.startswith("mmcblk") else
                "NVMe" if base.startswith("nvme") else
                "USB/SATA" if base.startswith("sd") else base)
        spin = " (rotational)" if rot == "1" else ""
        return f"{dev} [{kind}{spin}{' ' + model if model else ''}]"
    except Exception:
        return "unknown"


def thermal():
    """Pi-specific. A throttled run invalidates every timing in the file."""
    out = {}
    t = _cmd(["vcgencmd", "measure_temp"])
    if t:
        out["temp"] = t.replace("temp=", "")
    else:
        raw = _read("/sys/class/thermal/thermal_zone0/temp")
        if raw.isdigit():
            out["temp"] = f"{int(raw) / 1000:.1f}'C"
    g = _cmd(["vcgencmd", "get_throttled"])
    if g:
        code = g.split("=")[-1]
        out["throttled"] = code
        try:
            bits = int(code, 16)
            flags = []
            if bits & 0x1:      flags.append("under-voltage NOW")
            if bits & 0x2:      flags.append("freq capped NOW")
            if bits & 0x4:      flags.append("THROTTLED NOW")
            if bits & 0x10000:  flags.append("under-voltage occurred")
            if bits & 0x40000:  flags.append("throttling occurred")
            out["throttled_meaning"] = ", ".join(flags) or "clean"
        except ValueError:
            pass
    return out


def package_footprint(names=("numpy", "pandas", "pyarrow", "h5py", "duckdb")):
    """
    Installed size per library -- a real cost on a small device, and one
    data-engineering benchmarks never report.

    Sized from the distribution's file manifest rather than the package
    directory. Several of these ship their heavy shared object OUTSIDE the
    importable package: walking Path(module.__file__).parent reports duckdb
    at 0.9 MiB instead of ~53 MiB, because its binary lives elsewhere in
    site-packages.
    """
    import importlib.metadata as md

    sizes = {}
    for name in names:
        try:
            dist = md.distribution(name)
            base = Path(dist.locate_file(""))
            total = 0
            for f in (dist.files or []):
                try:
                    fp = base / f
                    if fp.is_file():
                        total += fp.stat().st_size
                except OSError:
                    pass
            sizes[name] = {"version": dist.version,
                           "mib": round(total / 1024**2, 1)}
        except Exception:
            sizes[name] = None
    return sizes


def environment(where=None):
    model = _read("/proc/device-tree/model")
    cpuinfo = _read("/proc/cpuinfo")
    if not model:
        m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
        model = m.group(1) if m else platform.processor() or "unknown"

    mem = ""
    m = re.search(r"^MemTotal:\s+(\d+) kB", _read("/proc/meminfo"), re.M)
    if m:
        mem = f"{int(m.group(1)) / 1024**2:.1f} GiB"

    osr = _read("/etc/os-release")
    distro = ""
    if osr:
        m = re.search(r'PRETTY_NAME="([^"]+)"', osr)
        distro = m.group(1) if m else ""

    return {
        "model": model,
        "machine": platform.machine(),
        "cores": os.cpu_count(),
        "memory": mem,
        "os": f"{platform.system()} {platform.release()}",
        "distro": distro,
        "python": platform.python_version(),
        "storage": root_storage(where),
        "packages": package_footprint(),
        "thermal_before": thermal(),
    }


def print_environment(env):
    print("=" * 74)
    print("  ENVIRONMENT")
    print("=" * 74)
    print(f"  device     {env['model']}")
    print(f"  cpu        {env['machine']}, {env['cores']} cores")
    print(f"  memory     {env['memory']}")
    print(f"  os         {env['distro'] or env['os']}")
    print(f"  python     {env['python']}")
    print(f"  storage    {env['storage']}")
    if env.get("thermal_before"):
        t = env["thermal_before"]
        print(f"  thermal    {t.get('temp', '?')}  "
              f"{t.get('throttled_meaning', '')}")
    pk = "  ".join(f"{k} {v['version']} ({v['mib']} MiB)"
                   for k, v in env["packages"].items() if v)
    print(f"  packages   {pk}")


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024
