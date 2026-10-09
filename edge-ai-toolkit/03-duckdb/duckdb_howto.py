"""
duckdb_howto.py -- using DuckDB over a canonical Parquet table.

    python duckdb_howto.py            run every demo
    python duckdb_howto.py --list     what each one shows
    python duckdb_howto.py validate   run just one

bench_duckdb.py answers "is it worth it". This file answers "what do I
type". Each demo is self-contained and prints what it did.

The through-line: post 1 made a constant impossible to get wrong by
generating it. This does the same thing one layer up -- it turns facts about
your dataset from things somebody verified once into assertions that run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent / "synth"), str(_HERE.parent), str(_HERE)]
from synth import generate  # noqa: E402

M = 1024 * 1024


def arrow_table(rel):
    """
    Get an Arrow table from a DuckDB result.

    fetch_arrow_table() was renamed to to_arrow_table() and now emits a
    DeprecationWarning. Both spellings need supporting, because the whole
    point of this repo is that it runs on whatever the reader has.
    """
    return (rel.to_arrow_table() if hasattr(rel, "to_arrow_table")
            else rel.fetch_arrow_table())


def hdr(title, why):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)
    print(f"  {why}\n")


def make_table(ws, sessions=60, seconds=60, seed_shift=0, subjects=12):
    df = generate(sessions, seconds, 50, subjects=subjects)
    return df.sort_values(["subject", "session", "t_ms"]).reset_index(drop=True)


# =========================================================================

def demo_query(ws):
    """
    Query a Parquet file without importing it anywhere.

    There is no load step, no schema declaration, no database file. The
    Parquet footer already contains the schema, so the engine reads it and
    starts answering questions.
    """
    hdr("1. Querying files in place",
        "No import step. The file already knows what is in it.")

    df = make_table(ws)
    p = ws / "canonical.parquet"
    df.to_parquet(p, index=False, compression="zstd")

    con = duckdb.connect()
    print(f"  {p.name}, {p.stat().st_size/M:.1f} MiB, {len(df):,} rows\n")

    rows = con.execute(f"""
        SELECT subject, count(*) AS n, count(DISTINCT session) AS sessions
        FROM read_parquet('{p}')
        GROUP BY subject ORDER BY subject LIMIT 5
    """).fetchall()
    print(f"  {'subject':>8}{'rows':>10}{'sessions':>10}")
    for r in rows:
        print(f"  {r[0]:>8}{r[1]:>10,}{r[2]:>10}")

    print("\n  DESCRIBE works too, straight off the footer:")
    for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')"
                         ).fetchall()[:4]:
        print(f"    {r[0]:<10} {r[1]}")
    print("    ...")


def demo_validate(ws):
    """
    Dataset validation as assertions rather than a script.

    These are the checks you would want to fail a build. Written as SQL they
    are short enough that people actually keep them up to date, and they run
    against the file rather than against a DataFrame someone remembered to
    build correctly.
    """
    hdr("2. Validation you can run in CI",
        "Facts about the dataset, as assertions instead of a notebook cell.")

    df = make_table(ws)
    # Introduce two realistic defects to prove the checks bite.
    df.loc[df["subject"] == 7, "label"] = 0          # a subject with no positives
    df.loc[df["session"] == 3, "gx"] = 0.0           # a dead channel in one session
    p = ws / "canonical.parquet"
    df.to_parquet(p, index=False, compression="zstd")

    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{p}')")

    checks = [
        ("no subject is missing entirely",
         "SELECT count(DISTINCT subject) FROM t", lambda v: v == 12),
        ("every subject has at least one positive example",
         """SELECT count(*) FROM (
                SELECT subject FROM t GROUP BY subject HAVING sum(label) = 0)""",
         lambda v: v == 0),
        ("no channel is constant across the whole table",
         """SELECT count(*) FROM (
                SELECT 1 FROM t HAVING min(gx) = max(gx))""",
         lambda v: v == 0),
        ("no session has a constant channel",
         """SELECT count(*) FROM (
                SELECT session FROM t GROUP BY session
                HAVING min(gx) = max(gx))""",
         lambda v: v == 0),
        ("class balance is within 20:1",
         "SELECT max(1.0*n/p) FROM (SELECT count(*) n, greatest(sum(label),1) p FROM t GROUP BY subject)",
         lambda v: v <= 20),
        ("no NULLs in any channel",
         "SELECT count(*) FROM t WHERE ax IS NULL OR gx IS NULL",
         lambda v: v == 0),
    ]

    failed = 0
    for name, sql, ok in checks:
        v = con.execute(sql).fetchone()[0]
        good = ok(v)
        failed += not good
        print(f"  {'ok  ' if good else 'FAIL'}  {name:<48} = {v}")

    print(f"\n  {len(checks)-failed}/{len(checks)} passed")
    print("\n  Three checks failed, and they were supposed to: this demo")
    print("  deliberately zeroes one subject's labels and flatlines one")
    print("  session's gx channel.")
    print("\n  Note that only TWO defects were injected. The class-balance")
    print("  check fails as a consequence of the first one, not")
    print("  independently -- a subject with no positives has an infinite")
    print("  ratio. Checks fail in correlated groups, so read the whole")
    print("  list before concluding you have three separate problems.")
    print("\n  A validation suite that has never failed tells you nothing.")
    print("  Same argument as the sabotage script in 01-contract: break it")
    print("  on purpose, and confirm it notices.")


def demo_leakage(ws):
    """
    Proving a train/test split is disjoint, in SQL.

    The safety-critical property in any subject-wise evaluation is that no
    subject appears on both sides. Written as a query it becomes something
    CI can assert rather than something a comment claims.
    """
    hdr("3. Asserting a split has no leakage",
        "The property your metrics depend on, checked instead of assumed.")

    df = make_table(ws)
    root = ws / "hive"
    pq.write_to_dataset(pa.Table.from_pandas(df, preserve_index=False),
                        root_path=str(root), partition_cols=["subject"],
                        compression="zstd")

    con = duckdb.connect()
    held_out = [3, 7]
    src = f"read_parquet('{root}/**/*.parquet', hive_partitioning = true)"

    overlap = con.execute(f"""
        WITH train AS (SELECT DISTINCT subject FROM {src}
                       WHERE subject NOT IN ({','.join(map(str, held_out))})),
             test  AS (SELECT DISTINCT subject FROM {src}
                       WHERE subject IN ({','.join(map(str, held_out))}))
        SELECT count(*) FROM train JOIN test USING (subject)
    """).fetchone()[0]

    counts = con.execute(f"""
        SELECT
          sum(CASE WHEN subject IN ({','.join(map(str, held_out))})
                   THEN 0 ELSE 1 END) AS train_rows,
          sum(CASE WHEN subject IN ({','.join(map(str, held_out))})
                   THEN 1 ELSE 0 END) AS test_rows
        FROM {src}
    """).fetchone()

    print(f"  held out subjects {held_out}")
    print(f"    train {counts[0]:>9,} rows")
    print(f"    test  {counts[1]:>9,} rows")
    print(f"    subjects on both sides: {overlap}   "
          f"{'(disjoint)' if overlap == 0 else '<-- LEAKAGE'}")

    print("\n  Because the data is partitioned by subject, the test query")
    print("  opens two directories and the train query opens the rest. The")
    print("  split is enforced by the filesystem layout, not by a boolean")
    print("  mask that could have the wrong sign.")


def demo_union(ws):
    """
    Querying two datasets with different columns as one table.

    The moment a second data source arrives is when a query engine starts
    earning its place. union_by_name matches on column name rather than
    position, and fills the gaps with NULL.
    """
    hdr("4. Two sources, one query",
        "union_by_name matches columns by name and NULLs the differences.")

    a = make_table(ws, sessions=30, subjects=6)
    a["source"] = "siteA"
    b = make_table(ws, sessions=30, subjects=6)
    b["source"] = "siteB"
    b["ambient_c"] = 21.5                     # a column site A never recorded

    (ws / "sources").mkdir()
    a.to_parquet(ws / "sources" / "siteA.parquet", index=False)
    b.to_parquet(ws / "sources" / "siteB.parquet", index=False)

    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT source,
               count(*)               AS n,
               count(DISTINCT subject) AS subjects,
               round(avg(ax), 4)      AS ax_mean,
               count(ambient_c)       AS has_ambient
        FROM read_parquet('{ws}/sources/*.parquet', union_by_name = true)
        GROUP BY source ORDER BY source
    """).fetchall()

    print(f"  {'source':<8}{'rows':>10}{'subjects':>10}{'ax_mean':>10}"
          f"{'ambient_c':>12}")
    for r in rows:
        print(f"  {r[0]:<8}{r[1]:>10,}{r[2]:>10}{r[3]:>10}{r[4]:>12,}")

    print("\n  siteA has no ambient_c column at all. Without union_by_name")
    print("  this query fails; with it, siteA's rows carry NULL there and")
    print("  count(ambient_c) reports 0 because count ignores NULLs.")
    print("\n  Comparing distributions per source like this is how you catch")
    print("  a second dataset that does not match the first -- different")
    print("  scaling, different orientation, different gravity convention.")


def demo_memory(ws):
    """
    Working under a memory budget.

    DuckDB streams in batches and spills to disk when it must, so you can
    give it a hard ceiling. pandas has no equivalent: it either materialises
    the table or the kernel kills the process.
    """
    hdr("5. A hard memory ceiling",
        "Give the engine a budget, and see what happens when it is too small.")

    df = make_table(ws, sessions=600, seconds=60)
    p = ws / "big.parquet"
    df.to_parquet(p, index=False, compression="zstd")
    mem = df.memory_usage(deep=True).sum() / M
    print(f"  {len(df):,} rows, {mem:.0f} MiB as a DataFrame, "
          f"{p.stat().st_size/M:.1f} MiB on disk")

    # A full sort written back out, not an aggregation and not a top-N.
    #
    # Both of those would prove nothing. An aggregation over twelve groups
    # needs almost no memory however large the input, and ORDER BY ... LIMIT
    # is rewritten into a top-N operator that never materialises the
    # relation. A sort of every row, with the result going to a file, is
    # work that genuinely has to hold the data.
    sql = (f"COPY (SELECT * FROM read_parquet('{p}') ORDER BY ax, t_ms) "
           f"TO '{ws}/sorted.parquet'")
    print(f"  query: sort every row by ax and write the result out\n")

    for limit in ("4GB", "256MB", "64MB", "16MB"):
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{limit}'")
        con.execute(f"SET temp_directory='{ws}/spill'")
        t0 = time.perf_counter()
        try:
            con.execute(sql)
            print(f"  memory_limit={limit:<7} completed in {time.perf_counter()-t0:5.2f}s")
        except Exception as e:
            print(f"  memory_limit={limit:<7} {type(e).__name__}")
            print(f"  {'':<21}{str(e).splitlines()[0][:52]}")
        con.close()
        shutil.rmtree(ws / "spill", ignore_errors=True)
        (ws / "sorted.parquet").unlink(missing_ok=True)

    print("\n  A budget smaller than the data still completes: the engine")
    print("  spills intermediate results to temp_directory and takes")
    print("  longer. (The spill files are deleted as soon as they are")
    print("  finished with, so there is nothing left to measure afterwards")
    print("  -- the time is the observable.)")
    print("\n  Below some point it stops, and this is the part that matters")
    print("  on a gateway: it raises an exception you can catch, rather")
    print("  than growing until the kernel's OOM killer takes the process")
    print("  out. pandas gives you the second behaviour.")


def demo_arrow(ws):
    """
    Handing results to training code without a copy.

    Arrow is a memory layout that numpy can view directly. Going out through
    Arrow rather than pandas avoids materialising an intermediate DataFrame.
    """
    hdr("6. Straight into numpy, via Arrow",
        "The last hop, from query result to model input.")

    df = make_table(ws, sessions=120)
    p = ws / "canonical.parquet"
    df.to_parquet(p, index=False, compression="zstd")
    con = duckdb.connect()

    sql = f"""SELECT ax, ay, az, gx, gy, gz FROM read_parquet('{p}')
              WHERE subject != 3"""

    t0 = time.perf_counter()
    pdf = con.execute(sql).df()
    t_df = time.perf_counter() - t0

    t0 = time.perf_counter()
    tbl = arrow_table(con.execute(sql))
    t_ar = time.perf_counter() - t0

    t0 = time.perf_counter()
    col = tbl.column("ax").combine_chunks().to_numpy(zero_copy_only=True)
    t_np = time.perf_counter() - t0

    print(f"  rows: {tbl.num_rows:,}\n")
    print(f"  .df()               -> pandas DataFrame   {t_df*1000:7.1f} ms")
    print(f"  .to_arrow_table()   -> arrow Table        {t_ar*1000:7.1f} ms")
    print(f"  arrow column -> numpy (zero copy)         {t_np*1000:7.1f} ms")
    print(f"\n  zero_copy_only=True succeeded, so that last step is a view")
    print(f"  over the same bytes: {col.dtype}, {col.nbytes/M:.1f} MiB not "
          f"copied.")
    print("\n  Worth knowing for post 4: this works because Arrow's in-memory")
    print("  layout is contiguous and typed the same way numpy's is.")


DEMOS = {
    "query":    demo_query,
    "validate": demo_validate,
    "leakage":  demo_leakage,
    "union":    demo_union,
    "memory":   demo_memory,
    "arrow":    demo_arrow,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, fn in DEMOS.items():
            first = [l.strip() for l in (fn.__doc__ or "").splitlines()
                     if l.strip()][0]
            print(f"  {name:<10}{first}")
        return

    unknown = [n for n in args.only if n not in DEMOS]
    if unknown:
        ap.error(f"unknown demo(s) {unknown}; choose from {list(DEMOS)}")
    chosen = args.only or list(DEMOS)

    print(f"  duckdb {duckdb.__version__}   pyarrow {pa.__version__}   "
          f"pandas {pd.__version__}")
    ws = Path(tempfile.mkdtemp(prefix="duck-howto-"))
    try:
        for name in chosen:
            sub = ws / name
            sub.mkdir()
            DEMOS[name](sub)
        print("\n" + "=" * 74)
        print(f"  {len(chosen)} demo(s) complete")
    finally:
        if args.keep:
            print(f"  scratch kept at {ws}")
        else:
            shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    main()
