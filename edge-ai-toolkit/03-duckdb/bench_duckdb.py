"""
bench_duckdb.py -- what does a query engine buy you over pandas?

Both read the same Parquet files. The question is not "is SQL nicer" -- it is
whether pushing the work into an engine that understands the file format
changes the time and the memory enough to matter on constrained hardware.

    python bench_duckdb.py                      quick, ~600k rows
    python bench_duckdb.py --rows 2400000 --repeat 5 \
        --json ../results/myhost-duckdb.json    the real run
    python bench_duckdb.py --list               what each task measures

METHOD
------
Same discipline as bench_formats.py, and for the same reasons:

  * every task runs in a FRESH SUBPROCESS, and peak memory is read from
    VmHWM rather than ru_maxrss, because ru_maxrss survives exec and a
    child otherwise reports its parent's peak
  * the page cache is dropped before each run, so reads are cold
  * median of --repeat runs
  * the environment, including thermal state, is reported alongside

The pandas side is written the way a careful person would write it, not
handicapped. Every task gives pandas `columns=` for the columns it actually
needs, and `filters=` where a predicate applies, because both exist and both
are the right way to use it.

That fairness is load-bearing. My first version of this benchmark omitted
`columns=` on four of the tasks and reported DuckDB at up to 123x. Most of
that gap was pandas materialising ten columns to count one -- 152 ms against
6 ms for the same answer, entirely within pandas. The `filtered_naive` task
preserves that comparison deliberately, because the cost of forgetting is
worth knowing, but it is labelled as what it is rather than being passed off
as an engine difference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent / "synth"), str(_HERE.parent), str(_HERE)]
from synth import generate  # noqa: E402
from common.measure import (  # noqa: E402
    peak_rss_mib, drop_cache, fsync_path, environment, print_environment,
    thermal, human,
)

M = 1024 * 1024
HELD_OUT = 3          # the subject we pretend to hold out for evaluation


# =========================================================================
# dataset preparation
# =========================================================================

def prepare(ws, rows, rate_hz=50, seconds=120):
    """
    Build the three shapes a real canonical table takes.

      single.parquet    one consolidated file, sorted by subject
      parts/            the same data as many small rollup files
      hive/             partitioned by subject, the honest-split layout

    Sorted by subject on purpose: post 2 showed that footer statistics only
    let a reader skip blocks when the data is laid out so their ranges do
    not overlap. Writing it shuffled here would measure a straw man.
    """
    sessions = max(2, int(rows / (rate_hz * seconds)))
    df = generate(sessions, seconds, rate_hz)
    df = df.sort_values(["subject", "session", "t_ms"]).reset_index(drop=True)

    single = ws / "single.parquet"
    df.to_parquet(single, index=False, compression="zstd")

    parts = ws / "parts"
    parts.mkdir()
    bounds = np.linspace(0, len(df), 25, dtype=int)
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        df.iloc[a:b].to_parquet(parts / f"part-{i:03d}.parquet",
                                index=False, compression="zstd")

    hive = ws / "hive"
    pq.write_to_dataset(pa.Table.from_pandas(df, preserve_index=False),
                        root_path=str(hive), partition_cols=["subject"],
                        compression="zstd")

    for p in (single, parts, hive):
        fsync_path(p)

    return df, {"single": single, "parts": parts, "hive": hive}


# =========================================================================
# the tasks
# =========================================================================
# Each is (description, pandas callable, duckdb callable). Both sides get
# the paths dict and must return something comparable so the harness can
# check they actually agree -- a faster wrong answer is not a result.

def t_full_aggregate():
    """Mean and std of every channel, grouped by subject. A full scan."""
    def pandas_(paths, con):
        # columns= matters enormously here; see t_filtered_naive below.
        df = pd.read_parquet(paths["single"],
                             columns=["subject", "ax", "ay", "az"])
        r = df.groupby("subject")[["ax", "ay", "az"]].agg(["mean", "std"])
        return float(r.to_numpy().sum())

    def duck_(paths, con):
        r = con.execute(f"""
            SELECT subject,
                   avg(ax), avg(ay), avg(az),
                   stddev_samp(ax), stddev_samp(ay), stddev_samp(az)
            FROM read_parquet('{paths["single"]}')
            GROUP BY subject ORDER BY subject
        """).fetchall()
        return float(sum(sum(x for x in row[1:]) for row in r))
    return pandas_, duck_


def t_one_column():
    """
    Average a single column out of ten.

    pandas gets the `columns=` argument, which is the right way to write it.
    This is the fairest possible version of the comparison.
    """
    def pandas_(paths, con):
        df = pd.read_parquet(paths["single"], columns=["ax"])
        return float(df["ax"].mean())

    def duck_(paths, con):
        return float(con.execute(
            f"SELECT avg(ax) FROM read_parquet('{paths['single']}')"
        ).fetchone()[0])
    return pandas_, duck_


def t_filtered_split():
    """
    The query an honest evaluation actually runs: everything except one
    held-out subject.
    """
    def pandas_(paths, con):
        df = pd.read_parquet(paths["single"], columns=["subject"],
                             filters=[("subject", "!=", HELD_OUT)])
        return int(len(df))

    def duck_(paths, con):
        return int(con.execute(f"""
            SELECT count(*) FROM read_parquet('{paths["single"]}')
            WHERE subject != {HELD_OUT}
        """).fetchone()[0])
    return pandas_, duck_


def t_hive_partition():
    """
    The same held-out query against a partitioned layout, where the answer
    is a directory the reader never opens.
    """
    def pandas_(paths, con):
        df = pd.read_parquet(paths["hive"], columns=["subject"],
                             filters=[("subject", "!=", HELD_OUT)])
        return int(len(df))

    def duck_(paths, con):
        return int(con.execute(f"""
            SELECT count(*) FROM read_parquet('{paths["hive"]}/**/*.parquet',
                                              hive_partitioning = true)
            WHERE subject != {HELD_OUT}
        """).fetchone()[0])
    return pandas_, duck_


def t_many_files():
    """
    Scan a folder of un-compacted rollup files, as a device that writes
    periodically would produce.
    """
    def pandas_(paths, con):
        df = pd.read_parquet(paths["parts"], columns=["subject", "label"])
        return int(df.groupby("subject")["label"].sum().sum())

    def duck_(paths, con):
        return int(con.execute(f"""
            SELECT sum(s) FROM (
              SELECT subject, sum(label) AS s
              FROM read_parquet('{paths["parts"]}/*.parquet')
              GROUP BY subject)
        """).fetchone()[0])
    return pandas_, duck_


def t_validation():
    """
    The real point of the post: a dataset validation suite.

    Class balance per subject, sample counts, which subjects carry no
    positive examples, and the min/max of every channel -- the checks you
    would want to run in CI before trusting any metric computed downstream.
    """
    def pandas_(paths, con):
        df = pd.read_parquet(
            paths["single"], columns=["subject", "label", "session", "ax"])
        g = df.groupby("subject")
        out = pd.DataFrame({
            "n": g.size(),
            "pos": g["label"].sum(),
            "sessions": g["session"].nunique(),
            "ax_min": g["ax"].min(),
            "ax_max": g["ax"].max(),
        })
        out["pos_frac"] = out["pos"] / out["n"]
        return int(out["n"].sum()), int((out["pos"] == 0).sum())

    def duck_(paths, con):
        r = con.execute(f"""
            SELECT subject,
                   count(*)                    AS n,
                   sum(label)                  AS pos,
                   count(DISTINCT session)     AS sessions,
                   min(ax) AS ax_min, max(ax)  AS ax_max,
                   sum(label) / count(*)       AS pos_frac
            FROM read_parquet('{paths["single"]}')
            GROUP BY subject ORDER BY subject
        """).fetchall()
        return int(sum(x[1] for x in r)), int(sum(1 for x in r if x[2] == 0))
    return pandas_, duck_


def t_to_arrow():
    """
    Handing the result to the training code.

    pandas materialises a DataFrame; DuckDB can emit an Arrow table that
    numpy can view without copying. Measures the whole path from file to
    something a model could consume.
    """
    def pandas_(paths, con):
        df = pd.read_parquet(paths["single"],
                             columns=["ax", "ay", "az", "gx", "gy", "gz"])
        return int(df.to_numpy(dtype="float32").shape[0])

    def duck_(paths, con):
        rel = con.execute(f"""
            SELECT ax, ay, az, gx, gy, gz
            FROM read_parquet('{paths["single"]}')
        """)
        tbl = (rel.to_arrow_table() if hasattr(rel, "to_arrow_table")
               else rel.fetch_arrow_table())
        return int(tbl.num_rows)
    return pandas_, duck_


def t_filtered_naive():
    """
    The same count, written the way it comes out first.

    Deliberately unfair, and the point of the task is exactly that.

    `filtered_split` gives pandas `columns=["subject"]`, because that is what
    the query needs. This one omits it, which is how most people write the
    line the first time -- there is nothing obviously wrong with
    `len(pd.read_parquet(path, filters=...))`.

    The gap between this task's pandas timing and `filtered_split`'s pandas
    timing is not a difference between engines. It is the cost of
    materialising ten columns to count one. On a Pi that was 152 ms against
    6 ms for the same answer.

    The engines differ in whether you have to REMEMBER. DuckDB works out
    which columns the query touches; pandas reads whatever you asked for,
    and by default you asked for everything.
    """
    def pandas_(paths, con):
        df = pd.read_parquet(paths["single"],
                             filters=[("subject", "!=", HELD_OUT)])
        return int(len(df))

    def duck_(paths, con):
        return int(con.execute(f"""
            SELECT count(*) FROM read_parquet('{paths["single"]}')
            WHERE subject != {HELD_OUT}
        """).fetchone()[0])
    return pandas_, duck_


TASKS = {
    "full_aggregate":  t_full_aggregate,
    "one_column":      t_one_column,
    "filtered_split":  t_filtered_split,
    "filtered_naive":  t_filtered_naive,
    "hive_partition":  t_hive_partition,
    "many_files":      t_many_files,
    "validation":      t_validation,
    "to_arrow":        t_to_arrow,
}


# =========================================================================
# worker
# =========================================================================

def results_agree(a, b, rel_tol=1e-6, abs_tol=1e-6):
    """
    Do the two engines agree, allowing for float accumulation order?

    Exact equality is the wrong test, for the same reason the golden vectors
    in 01-contract use tolerances: two correct implementations that sum a
    column in different orders will not produce identical floats. pandas and
    DuckDB differ by about 1e-8 relative on a mean over 300,000 float32
    values, and float32 epsilon is 1.2e-7.

    BOTH tolerances are needed, and the second one is easy to forget. A
    relative test alone fails whenever the true answer happens to sit near
    zero: on one run here the mean of a channel came out at -0.0022, the two
    engines differed by 3.6e-9 in absolute terms, and that is 1.6e-6
    relative -- a spurious failure caused entirely by where the answer
    landed rather than by anything either engine did.

    Integers still have to match exactly. A row count is a row count.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(
            results_agree(x, y, rel_tol, abs_tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(a, int):
        return a == b
    if isinstance(a, float):
        if a != a or b != b:                 # NaN never agrees
            return False
        diff = abs(a - b)
        return diff <= abs_tol or diff <= rel_tol * max(abs(a), abs(b))
    return a == b


def run_worker(task, engine, ws, repeat, memory_limit=None):
    paths = {"single": ws / "single.parquet",
             "parts": ws / "parts", "hive": ws / "hive"}
    pandas_fn, duck_fn = TASKS[task]()

    con = None
    if engine == "duckdb":
        import duckdb
        con = duckdb.connect()
        # Single-threading DuckDB would flatter pandas unfairly; leave the
        # default parallelism alone and record the core count instead.
        if memory_limit:
            con.execute(f"SET memory_limit='{memory_limit}'")

    fn = pandas_fn if engine == "pandas" else duck_fn

    # Peak after imports and connection setup, before any data is touched.
    # pandas and duckdb have different import costs, so a bare difference in
    # absolute peak would partly just be measuring `import`.
    baseline = peak_rss_mib()

    times, result = [], None
    for _ in range(repeat):
        for p in paths.values():
            drop_cache(p)
        t0 = time.perf_counter()
        result = fn(paths, con)
        times.append(time.perf_counter() - t0)

    print("@@RESULT@@" + json.dumps({
        "seconds": statistics.median(times),
        "all": times,
        "peak_rss_mib": peak_rss_mib(),
        "rss_baseline_mib": baseline,
        "rss_delta_mib": max(0.0, peak_rss_mib() - baseline),
        "result": result if isinstance(result, (int, float, list)) else
                  list(result) if isinstance(result, tuple) else repr(result),
    }))


# =========================================================================
# orchestrator
# =========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=600_000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--memory-limit", default=None,
                    help="cap DuckDB's memory, e.g. 256MB, to show spilling")
    ap.add_argument("--scaling", action="store_true",
                    help="sweep dataset sizes and report how peak RAM grows")
    ap.add_argument("--tmpdir")
    ap.add_argument("--json")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--worker")
    ap.add_argument("--engine")
    ap.add_argument("--ws")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.worker, args.engine, Path(args.ws), args.repeat,
                   args.memory_limit)
        return

    if args.list:
        for name, fn in TASKS.items():
            doc = [l.strip() for l in (fn.__doc__ or "").splitlines()
                   if l.strip()][0]
            print(f"  {name:<18}{doc}")
        return

    try:
        import duckdb
    except ImportError:
        raise SystemExit("duckdb is not installed:  pip install duckdb")

    chosen = args.only or list(TASKS)
    unknown = [t for t in chosen if t not in TASKS]
    if unknown:
        ap.error(f"unknown task(s) {unknown}; choose from {list(TASKS)}")

    ws = Path(args.tmpdir) if args.tmpdir else Path(
        tempfile.mkdtemp(prefix="duckbench-"))
    ws.mkdir(parents=True, exist_ok=True)

    env = environment(ws)
    print_environment(env)
    print(f"  duckdb     {duckdb.__version__}")
    print()

    if args.scaling:
        return run_scaling(args, ws, env)

    print(f"  preparing {args.rows:,} rows in {ws} ...", flush=True)
    t0 = time.perf_counter()
    df, paths = prepare(ws, args.rows)
    print(f"  {len(df):,} rows, {df.memory_usage(deep=True).sum()/M:.1f} MiB "
          f"in memory   ({time.perf_counter()-t0:.1f} s)")
    print(f"  single {human(paths['single'].stat().st_size)}   "
          f"parts {len(list(paths['parts'].glob('*.parquet')))} files   "
          f"hive {len(list(paths['hive'].iterdir()))} partitions")
    if args.memory_limit:
        print(f"  duckdb memory_limit = {args.memory_limit}")
    print()

    results = []
    hdr = (f"  {'task':<18}{'pandas':>9}{'duckdb':>9}{'speedup':>9}"
           f"{'pd RAM+':>9}{'db RAM+':>9}{'RAM ratio':>11}  agree")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for task in chosen:
        row = {"task": task}
        for engine in ("pandas", "duckdb"):
            cmd = [sys.executable, __file__, "--worker", task,
                   "--engine", engine, "--ws", str(ws),
                   "--repeat", str(args.repeat)]
            if engine == "duckdb" and args.memory_limit:
                cmd += ["--memory-limit", args.memory_limit]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            mark = [l for l in proc.stdout.splitlines()
                    if l.startswith("@@RESULT@@")]
            if not mark:
                row[engine] = {"error": (proc.stderr or proc.stdout or
                                         "no output").strip().splitlines()[-1]}
                continue
            row[engine] = json.loads(mark[0][len("@@RESULT@@"):])

        p, d = row.get("pandas", {}), row.get("duckdb", {})
        if "error" in p or "error" in d:
            print(f"  {task:<18}FAILED: "
                  f"{p.get('error') or d.get('error')}")
            results.append(row)
            continue

        agree = results_agree(p["result"], d["result"])
        row["agree"] = agree
        print(f"  {task:<18}{p['seconds']:>8.2f}s{d['seconds']:>8.2f}s"
              f"{p['seconds']/d['seconds']:>8.1f}x"
              f"{p['rss_delta_mib']:>8.0f}M{d['rss_delta_mib']:>8.0f}M"
              f"{p['rss_delta_mib']/max(d['rss_delta_mib'], 1):>10.2f}x"
              f"   {'yes' if agree else 'NO'}")
        results.append(row)

    disagreed = [r["task"] for r in results if r.get("agree") is False]
    if disagreed:
        print(f"\n  !! the two engines returned DIFFERENT answers for: "
              f"{', '.join(disagreed)}")
        print("     A faster wrong answer is not a result. Investigate "
              "before quoting any timing above.")

    after = thermal()
    if after:
        print(f"\n  thermal after: {after.get('temp','?')}  "
              f"{after.get('throttled_meaning','')}")
        if "occurred" in after.get("throttled_meaning", ""):
            print("  !! throttling occurred -- timings are not trustworthy.")

    if args.json:
        env["thermal_after"] = after
        Path(args.json).write_text(json.dumps(
            {"environment": env, "duckdb_version": duckdb.__version__,
             "args": vars(args), "rows": int(len(df)),
             "results": results}, indent=2))
        print(f"\n  wrote {args.json}")

    if not args.tmpdir:
        shutil.rmtree(ws, ignore_errors=True)


def run_scaling(args, ws, env):
    """
    How does peak memory grow with dataset size, for each engine?

    This is the honest way to answer "what happens when the data does not
    fit in RAM" without generating a file larger than the machine. If pandas
    grows linearly and DuckDB stays flat, the extrapolation to the point
    where pandas fails is straightforward and you can state it as one.
    """
    import duckdb
    sizes = [args.rows // 4, args.rows // 2, args.rows, args.rows * 2]
    print(f"  {'rows':>12}{'pd time':>10}{'db time':>10}"
          f"{'pd RAM':>10}{'db RAM':>10}")
    print("  " + "-" * 52)

    out = []
    for n in sizes:
        sub = ws / f"scale_{n}"
        sub.mkdir(exist_ok=True)
        prepare(sub, n)
        row = {"rows": n}
        for engine in ("pandas", "duckdb"):
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", "full_aggregate",
                 "--engine", engine, "--ws", str(sub),
                 "--repeat", str(max(1, args.repeat // 2))],
                capture_output=True, text=True)
            mark = [l for l in proc.stdout.splitlines()
                    if l.startswith("@@RESULT@@")]
            row[engine] = json.loads(mark[0][len("@@RESULT@@"):]) if mark \
                else {"error": (proc.stderr or "?").strip().splitlines()[-1]}
        p, d = row["pandas"], row["duckdb"]
        if "error" in p or "error" in d:
            print(f"  {n:>12,}  FAILED: {p.get('error') or d.get('error')}")
        else:
            print(f"  {n:>12,}{p['seconds']:>9.2f}s{d['seconds']:>9.2f}s"
                  f"{p['peak_rss_mib']:>9.0f}M{d['peak_rss_mib']:>9.0f}M")
        out.append(row)
        shutil.rmtree(sub, ignore_errors=True)

    print("\n  Watch the two RAM columns rather than the times. pandas must")
    print("  materialise the table; DuckDB streams it in batches. Where the")
    print("  first column grows with the data and the second does not, the")
    print("  point at which pandas stops fitting is arithmetic.")

    if args.json:
        env["thermal_after"] = thermal()
        Path(args.json).write_text(json.dumps(
            {"environment": env, "duckdb_version": duckdb.__version__,
             "mode": "scaling", "args": vars(args), "results": out}, indent=2))
        print(f"\n  wrote {args.json}")
    shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    main()
