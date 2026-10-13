"""
bench_formats.py -- what does each storage format actually cost?

Measures, for the same canonical sensor table:

    file size      on disk, after compression
    write time     seconds to serialise
    read time      seconds to load back, with the page cache dropped first
    peak memory    peak RSS of a fresh process doing only that read
    dtype fidelity does the round-trip return the schema you wrote?

METHOD NOTES -- these are the things that make benchmark numbers wrong, and
they are worth repeating in any write-up that quotes the results.

  * Every read runs in a FRESH SUBPROCESS. Peak RSS is a high-water mark for
    the life of a process, so measuring several formats in one process
    reports the largest one for all of them.

  * The page cache is dropped per file with posix_fadvise(DONTNEED) before
    each read. Without it the second read of a file is served from RAM and
    you are timing memory bandwidth, not the format. This needs no root.

  * Each measurement runs --repeat times and reports the MEDIAN. A single
    timing on an SD card is noise.

  * On a Raspberry Pi, check the thermal report at the bottom of the output.
    A throttled run produces slow numbers that have nothing to do with the
    formats being compared.

USAGE
    python bench_formats.py                       # quick, ~600k rows
    python bench_formats.py --sessions 400        # the real run, ~2.4M rows
    python bench_formats.py --skip json           # json is slow and huge
    python bench_formats.py --json results.json   # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# synth.py lives at the repo root because posts 2, 3 and 4 all build on the
# same dataset. Falls back to this directory so the file still runs if it is
# copied out on its own.
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent / "synth"), str(_HERE.parent), str(_HERE)]
from synth import generate, describe, DTYPES  # noqa: E402
from common.measure import (  # noqa: E402
    peak_rss_mib, drop_cache, environment, print_environment, thermal, human,
)


# =========================================================================
# format registry
# =========================================================================
# Each entry: (extension, write callable, read callable).
# Readers must return a DataFrame so fidelity can be compared column by column.

def _w_csv(df, p):      df.to_csv(p, index=False)
def _r_csv(p):          return pd.read_csv(p)


def _w_csv_gz(df, p):   df.to_csv(p, index=False, compression="gzip")
def _r_csv_gz(p):       return pd.read_csv(p, compression="gzip")


def _w_json(df, p):     df.to_json(p, orient="records", lines=True)
def _r_json(p):         return pd.read_json(p, orient="records", lines=True)


def _w_pickle(df, p):   df.to_pickle(p)
def _r_pickle(p):       return pd.read_pickle(p)


def _w_npz(df, p):
    np.savez_compressed(p, **{c: df[c].to_numpy() for c in df.columns})


def _r_npz(p):
    with np.load(p) as z:
        return pd.DataFrame({c: z[c] for c in z.files})


def _w_hdf5(df, p):
    import h5py
    with h5py.File(p, "w") as f:
        for c in df.columns:
            f.create_dataset(c, data=df[c].to_numpy(),
                             compression="gzip", compression_opts=4)


def _r_hdf5(p):
    import h5py
    with h5py.File(p, "r") as f:
        return pd.DataFrame({c: f[c][:] for c in f.keys()})


def _parquet(compression):
    def w(df, p):
        df.to_parquet(p, index=False, compression=compression, engine="pyarrow")
    return w


def _r_parquet(p):      return pd.read_parquet(p, engine="pyarrow")


def _w_feather(df, p):
    import pyarrow.feather as feather
    feather.write_feather(df, p, compression="zstd")


def _r_feather(p):
    import pyarrow.feather as feather
    return feather.read_feather(p)


FORMATS = {
    "csv":            (".csv",     _w_csv,               _r_csv),
    "csv.gz":         (".csv.gz",  _w_csv_gz,            _r_csv_gz),
    "json":           (".jsonl",   _w_json,              _r_json),
    "pickle":         (".pkl",     _w_pickle,            _r_pickle),
    "npz":            (".npz",     _w_npz,               _r_npz),
    "hdf5":           (".h5",      _w_hdf5,              _r_hdf5),
    "parquet-none":   (".parquet", _parquet(None),       _r_parquet),
    "parquet-snappy": (".parquet", _parquet("snappy"),   _r_parquet),
    "parquet-zstd":   (".parquet", _parquet("zstd"),     _r_parquet),
    "feather-zstd":   (".feather", _w_feather,           _r_feather),
}

# Formats whose libraries may be absent. Skipped with a note, not an error.
OPTIONAL = {"hdf5": "h5py",
            "parquet-none": "pyarrow", "parquet-snappy": "pyarrow",
            "parquet-zstd": "pyarrow", "feather-zstd": "pyarrow"}


# =========================================================================
# measurement plumbing
# =========================================================================

def write_and_sync(writer, df, path, sync=True):
    """
    Serialise, then force the bytes to the device, and time both.

    Without the fsync, a write is timed only as far as the page cache. That
    is not a small effect: pickle appeared to write 88 MB in 0.043 s, which
    is over 2 GB/s and faster than any SD card can physically accept. The
    kernel had simply accepted the pages and returned.

    Timing to durability is the honest comparison, because the number people
    care about is "how long until my data is safely on the card".
    """
    t0 = time.perf_counter()
    writer(df, path)
    if not sync:
        # --no-fsync: reproduce the original mistake on purpose. The time
        # stops as soon as the kernel has accepted the bytes, before they
        # reach the card. Use it only to see how wrong the numbers get.
        return time.perf_counter() - t0
    # Directory fsync too, so the metadata entry is durable, not just data.
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return time.perf_counter() - t0


def fidelity(df, original_dtypes):
    """
    Did the round-trip preserve the declared schema?

    This is the measurement that matters most and the one nobody publishes.
    A format that silently widens int8 to int64 and float32 to float64 has
    not stored your data -- it has stored a lossy transcription of it that
    happens to hold the same values.
    """
    got, lost = {}, []
    for col, want in original_dtypes.items():
        if col not in df.columns:
            lost.append(f"{col}: MISSING")
            continue
        have = str(df[col].dtype)
        got[col] = have
        if have != want:
            lost.append(f"{col}: {want} -> {have}")
    return {"dtypes": got, "changed": lost, "exact": not lost}


# =========================================================================
# worker: one format, one fresh process
# =========================================================================

def run_worker(fmt, path, repeat, declared):
    ext, _, reader = FORMATS[fmt]

    # Baseline: interpreter plus the imports, before any data is touched.
    # Reporting absolute peak RSS would mostly report pandas' import cost,
    # which is identical for every format and drowns the difference being
    # measured. The delta is the number that means something.
    base = peak_rss_mib()

    times = []
    df = None
    for _ in range(repeat):
        drop_cache(path)
        t0 = time.perf_counter()
        df = reader(path)
        times.append(time.perf_counter() - t0)

    out = {
        "read_s": statistics.median(times),
        "read_s_all": times,
        "rss_baseline_mib": base,
        "rss_peak_mib": peak_rss_mib(),
        "rss_delta_mib": max(0.0, peak_rss_mib() - base),
        "rows": len(df),
        "fidelity": fidelity(df, declared),
    }
    print("@@RESULT@@" + json.dumps(out))


# =========================================================================
# environment report
# =========================================================================

# =========================================================================
# orchestrator
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--rate-hz", type=float, default=50)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="only these formats")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these formats")
    ap.add_argument("--raw-ticks", action="store_true",
                    help="store int16 ticks instead of scaled float32")
    ap.add_argument("--tmpdir", help="where to write test files "
                                     "(default: system temp)")
    ap.add_argument("--json", help="also write results here")
    ap.add_argument("--no-fsync", action="store_true",
                    help="time writes without fsync, to reproduce the "
                         "measurement mistake described in the series "
                         "(results are wrong on purpose)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--path", help=argparse.SUPPRESS)
    ap.add_argument("--declared", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:                       # child process: measure one read
        declared = json.loads(args.declared) if args.declared else dict(DTYPES)
        run_worker(args.worker, args.path, args.repeat, declared)
        return

    chosen = [f for f in (args.only or FORMATS) if f not in args.skip]
    unknown = [f for f in chosen if f not in FORMATS]
    if unknown:
        ap.error(f"unknown format(s): {unknown}. "
                 f"available: {', '.join(FORMATS)}")

    env = environment()
    print("=" * 74)
    print("  ENVIRONMENT")
    print("=" * 74)
    print(f"  device     {env['model']}")
    print(f"  cpu        {env['machine']}, {env['cores']} cores")
    print(f"  memory     {env['memory']}")
    print(f"  os         {env['distro'] or env['os']}")
    print(f"  python     {env['python']}")
    print(f"  storage    {env['storage']}")
    if env["thermal_before"]:
        t = env["thermal_before"]
        print(f"  thermal    {t.get('temp', '?')}  "
              f"{t.get('throttled_meaning', '')}")
    print("  packages   " + "  ".join(
        f"{k} {v['version']} ({v['mib']} MiB)"
        for k, v in env["packages"].items() if v))

    print()
    print("  generating dataset...", flush=True)
    t0 = time.perf_counter()
    df = generate(args.sessions, args.seconds, args.rate_hz,
                  scaled=not args.raw_ticks)
    gen_s = time.perf_counter() - t0
    info = describe(df)
    print(f"  {info['rows']:,} rows x {info['columns']} cols, "
          f"{info['sessions']} sessions, {info['in_memory_mib']} MiB in RAM "
          f"({gen_s:.1f} s)")
    declared = dict(DTYPES)
    if args.raw_ticks:
        declared.update({c: "int16" for c in
                         ("ax", "ay", "az", "gx", "gy", "gz")})

    tmp = Path(args.tmpdir) if args.tmpdir else Path(tempfile.mkdtemp(
        prefix="fmtbench-"))
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"  scratch    {tmp}")
    if args.no_fsync:
        print("  WARNING    --no-fsync: write times stop at the page cache and")
        print("             are wrong on purpose. Do not publish these numbers.")
    free = shutil.disk_usage(tmp).free
    print(f"  free space {human(free)}")
    print()

    results = []
    for fmt in chosen:
        ext, writer, _ = FORMATS[fmt]
        need = OPTIONAL.get(fmt)
        if need:
            try:
                __import__(need)
            except ImportError:
                print(f"  {fmt:<16} SKIPPED ({need} not installed)")
                continue

        path = tmp / f"bench{ext}"
        row = {"format": fmt}
        try:
            wt = []
            for _ in range(args.repeat):
                if path.exists():
                    path.unlink()
                wt.append(write_and_sync(writer, df, path,
                                         sync=not args.no_fsync))
            row["write_s"] = statistics.median(wt)
            row["write_s_all"] = wt
            row["bytes"] = path.stat().st_size

            proc = subprocess.run(
                [sys.executable, __file__, "--worker", fmt,
                 "--path", str(path), "--repeat", str(args.repeat),
                 "--declared", json.dumps(declared)],
                capture_output=True, text=True)
            marker = [ln for ln in proc.stdout.splitlines()
                      if ln.startswith("@@RESULT@@")]
            if not marker:
                raise RuntimeError(
                    (proc.stderr or proc.stdout or "no output").strip()
                    .splitlines()[-1])
            row.update(json.loads(marker[0][len("@@RESULT@@"):]))

            print(f"  {fmt:<16} {human(row['bytes']):>12}  "
                  f"w {row['write_s']:6.2f}s  r {row['read_s']:6.2f}s  "
                  f"+{row['rss_delta_mib']:5.0f} MiB  "
                  f"{'exact' if row['fidelity']['exact'] else 'DTYPES CHANGED'}")
        except Exception as e:
            row["error"] = str(e)
            print(f"  {fmt:<16} FAILED: {e}")
        finally:
            if path.exists():
                path.unlink()
        results.append(row)

    ok = [r for r in results if "bytes" in r]
    baseline = next((r["bytes"] for r in ok if r["format"] == "csv"), None)
    raw = info["in_memory_bytes"]

    print()
    print("=" * 74)
    print("  RESULTS")
    print("=" * 74)
    print(f"  in memory (pandas): {human(raw)}")
    print()
    hdr = (f"  {'format':<16}{'size':>11}{'vs csv':>9}{'write':>8}"
           f"{'read':>8}{'peak RAM':>10}  dtypes")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted(ok, key=lambda x: x["bytes"]):
        ratio = f"{baseline / r['bytes']:.1f}x" if baseline else "-"
        fid = "exact" if r["fidelity"]["exact"] else \
              ", ".join(r["fidelity"]["changed"][:2]) + \
              ("..." if len(r["fidelity"]["changed"]) > 2 else "")
        print(f"  {r['format']:<16}{human(r['bytes']):>11}{ratio:>9}"
              f"{r['write_s']:>7.2f}s{r['read_s']:>7.2f}s"
              f"{r['rss_peak_mib']:>9.0f}M  {fid}")

    bases = [r["rss_baseline_mib"] for r in ok if "rss_baseline_mib" in r]
    if bases:
        print()
        print(f"  peak RAM is for a fresh process doing only that read. "
              f"A bare interpreter")
        print(f"  with the same imports already costs "
              f"~{statistics.median(bases):.0f} MiB, so treat that as the "
              f"floor --")
        print(f"  differences below it are not meaningful, and a format only "
              f"stands out")
        print(f"  once its read pushes past it.")

    changed = [r for r in ok if not r["fidelity"]["exact"]]
    if changed:
        print()
        print("  SCHEMA NOT PRESERVED")
        for r in changed:
            print(f"    {r['format']}")
            for c in r["fidelity"]["changed"]:
                print(f"      {c}")

    after = thermal()
    if after:
        print()
        print(f"  thermal after: {after.get('temp', '?')}  "
              f"{after.get('throttled_meaning', '')}")
        if "occurred" in after.get("throttled_meaning", ""):
            print("  !! throttling occurred during this run -- timings are "
                  "not trustworthy.")
            print("     Cool the board, then re-run before quoting numbers.")

    if args.json:
        env["thermal_after"] = after
        Path(args.json).write_text(json.dumps(
            {"environment": env, "dataset": info,
             "args": vars(args), "results": results}, indent=2))
        print(f"\n  wrote {args.json}")

    try:
        if not args.tmpdir:
            shutil.rmtree(tmp)
    except OSError:
        pass

    # A format that failed to write or read is an error, so exit with one,
    # and a CI job running this benchmark fails. Formats skipped because
    # their library is not installed are not counted here.
    failed = [r["format"] for r in results if "error" in r]
    if failed:
        print(f"\n  !! these formats failed: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
