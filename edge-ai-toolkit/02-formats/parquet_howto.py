"""
parquet_howto.py -- how to actually use Parquet for edge AI training data.

    python parquet_howto.py              run every demo
    python parquet_howto.py --list       what each one shows
    python parquet_howto.py schema       run just one

Each demo is self-contained, prints what it did, and cleans up after itself.
Nothing here needs hardware, a sensor, or a downloaded dataset -- the data
comes from ../synth.

The benchmark next door (bench_formats.py) answers "which format". This file
answers "now what do I type".
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent / "synth"), str(_HERE)]
from synth import generate, CHANNELS  # noqa: E402

M = 1024 * 1024


def hdr(title, why):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)
    print(f"  {why}\n")


def size(p):
    p = Path(p)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


# =========================================================================

def demo_write(ws):
    """
    Write a canonical table and prove the schema survived.

    The habit worth forming: declare the dtypes you mean, then VERIFY the
    round-trip rather than assuming it. Half the formats in the benchmark
    silently widen int8 to int64 and float32 to float64, and the widening is
    invisible until something runs out of memory.
    """
    hdr("1. Writing, and checking the schema came back",
        "Declare dtypes deliberately. Then verify -- do not assume.")

    df = generate(40, 60, 50)
    before = {c: str(t) for c, t in df.dtypes.items()}

    pqt, csv = ws / "canonical.parquet", ws / "canonical.csv"
    df.to_parquet(pqt, index=False, compression="zstd")
    df.to_csv(csv, index=False)

    print(f"  {len(df):,} rows, {df.memory_usage(deep=True).sum()/M:.1f} MiB "
          f"in memory\n")
    print(f"  {'column':<10}{'declared':<11}{'parquet':<11}{'csv'}")
    print("  " + "-" * 44)
    back_pq = pd.read_parquet(pqt)
    back_csv = pd.read_csv(csv)
    for c in df.columns:
        a, b = str(back_pq[c].dtype), str(back_csv[c].dtype)
        flag = "" if b == before[c] else "  <- widened"
        print(f"  {c:<10}{before[c]:<11}{a:<11}{b}{flag}")

    print(f"\n  parquet {size(pqt)/M:6.1f} MiB   csv {size(csv)/M:6.1f} MiB")
    print(f"  csv round-trip in RAM: "
          f"{back_csv.memory_usage(deep=True).sum()/M:.1f} MiB vs "
          f"{back_pq.memory_usage(deep=True).sum()/M:.1f} MiB for parquet")
    print("\n  The CSV holds the same VALUES. It does not hold the same")
    print("  TABLE -- every width you chose was thrown away and guessed")
    print("  again on read.")


def demo_schema(ws):
    """
    Read the schema without reading the data.

    Parquet writes a footer holding the schema and per-column statistics.
    Both are reachable by seeking to the end of the file, so you can audit a
    file you were handed without loading it, and without executing anything
    it contains.
    """
    hdr("2. Inspecting a file without opening the data",
        "The footer is the point. You can audit a file you did not write.")

    df = generate(200, 120, 50)
    p = ws / "big.parquet"
    df.to_parquet(p, index=False, compression="zstd")

    t0 = time.perf_counter()
    schema = pq.read_schema(p)
    meta = pq.read_metadata(p)
    meta_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pd.read_parquet(p)
    full_s = time.perf_counter() - t0

    print(f"  file {size(p)/M:.1f} MiB, {meta.num_rows:,} rows\n")
    print(f"  read_schema + read_metadata : {meta_s*1000:8.1f} ms")
    print(f"  read the whole table        : {full_s*1000:8.1f} ms")
    print(f"  ratio                       : {full_s/meta_s:8.0f}x\n")
    print("  schema:")
    for name, typ in zip(schema.names, schema.types):
        print(f"    {name:<10} {typ}")
    print(f"\n  row groups: {meta.num_row_groups}   "
          f"created by: {meta.created_by}")
    print("\n  Compare with a pickle: to learn what is inside one, you have")
    print("  to unpickle it, and unpickling runs whatever it says to run.")


def demo_internals(ws):
    """
    Look at what Parquet actually did to the data.

    This is the mechanism behind the benchmark's headline result: an
    uncompressed Parquet file is far smaller than the in-memory layout,
    before any compression codec is involved. The encoding does that.
    """
    hdr("3. What Parquet does before any compression",
        "Dictionary and run-length encoding, per column, in the file itself.")

    df = generate(60, 120, 50)
    p = ws / "internals.parquet"
    df.to_parquet(p, index=False, compression=None)      # no codec at all

    meta = pq.read_metadata(p)
    rg = meta.row_group(0)

    print(f"  {len(df):,} rows, {df.memory_usage(deep=True).sum()/M:.1f} MiB "
          f"in memory")
    print(f"  on disk with NO compression codec: {size(p)/M:.1f} MiB\n")

    print(f"  {'column':<10}{'physical':<10}{'distinct':>10}"
          f"{'uncompressed':>14}{'dict?':>7}")
    print("  " + "-" * 53)
    for i in range(rg.num_columns):
        c = rg.column(i)
        name = c.path_in_schema
        print(f"  {name:<10}{str(c.physical_type):<10}"
              f"{df[name].nunique():>10,}"
              f"{c.total_uncompressed_size/1024:>12.0f} K"
              f"{'yes' if c.dictionary_page_offset else 'no':>7}")

    print("\n  'session' repeats for thousands of consecutive rows, so it")
    print("  costs almost nothing. The channels hold a few tens of thousands")
    print("  of distinct values out of millions of samples, because the ADC")
    print("  quantised them -- so they dictionary-encode too.")
    print("\n  Note the physical types. Parquet has no 16-bit type: int16 is")
    print("  stored as INT32 with a logical annotation. That is why storing")
    print("  raw int16 ticks instead of scaled float32 saves ~22% in every")
    print("  other format and ~0% here.")


def demo_rowgroups(ws):
    """
    Row groups, and why the default may not suit you.

    A row group is the unit of parallelism and of skipping. Each one carries
    min/max statistics per column in the footer, so a reader can decide not
    to decompress it at all.
    """
    hdr("4. Row groups: the unit of skipping",
        "Footer statistics let a reader skip a block without decoding it.")

    df = generate(200, 120, 50)

    print(f"  {len(df):,} rows\n")
    print(f"  {'row_group_size':>16}{'groups':>9}{'MiB':>8}"
          f"{'read all':>11}{'read 1 col':>12}")
    print("  " + "-" * 56)

    for rgs in (None, 200_000, 50_000):
        p = ws / f"rg_{rgs}.parquet"
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(tbl, p, compression="zstd",
                       **({"row_group_size": rgs} if rgs else {}))
        meta = pq.read_metadata(p)

        t0 = time.perf_counter()
        pq.read_table(p)
        t_all = time.perf_counter() - t0

        t0 = time.perf_counter()
        pq.read_table(p, columns=["ax"])
        t_one = time.perf_counter() - t0

        label = "default" if rgs is None else f"{rgs:,}"
        print(f"  {label:>16}{meta.num_row_groups:>9}{size(p)/M:>8.1f}"
              f"{t_all*1000:>9.0f}ms{t_one*1000:>10.0f}ms")
        p.unlink()

    print("\n  The default is 1,048,576 rows, which for a table this size")
    print("  gives one large group and one small remainder. Smaller groups")
    print("  mean finer skipping and more metadata; larger groups compress")
    print("  slightly better. Tune it only if you have a reason.")
    print("\n  Reading one column out of ten is the free win. It costs a")
    print("  fraction of the full read because the other nine columns are")
    print("  never touched -- that is what 'columnar' buys you day to day.")


def demo_stats(ws):
    """
    The min/max statistics that make skipping possible.

    Sorting before writing is what turns footer statistics from decoration
    into a usable index. The measure that matters is not whether ranges
    technically overlap, but how many row groups a real query has to open.
    """
    hdr("5. Footer statistics: sorted data can be skipped",
        "How many row groups must a reader actually touch?")

    df = generate(120, 60, 50)
    target = 42                        # pretend we want one session

    for label, frame in (
            ("sorted by session", df.sort_values("session")
             .reset_index(drop=True)),
            ("shuffled", df.sample(frac=1.0, random_state=0)
             .reset_index(drop=True))):

        p = ws / "stats.parquet"
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False),
                       p, compression="zstd", row_group_size=50_000)
        meta = pq.read_metadata(p)
        idx = list(frame.columns).index("session")

        # Exactly what a reader does: check each row group's min/max against
        # the predicate, and skip any group that cannot contain a match.
        must_open = []
        for i in range(meta.num_row_groups):
            st = meta.row_group(i).column(idx).statistics
            if st.min <= target <= st.max:
                must_open.append(i)

        print(f"  {label}")
        for i in range(min(meta.num_row_groups, 3)):
            st = meta.row_group(i).column(idx).statistics
            print(f"    row group {i}: session {st.min:>4} .. {st.max:>4}")
        if meta.num_row_groups > 3:
            print(f"    ... {meta.num_row_groups - 3} more")
        print(f"    -> WHERE session = {target} must open "
              f"{len(must_open)} of {meta.num_row_groups} row groups "
              f"({len(must_open) / meta.num_row_groups:.0%})\n")
        p.unlink()

    print("  Same data, same size, same format. The only difference is the")
    print("  order it was written in.")
    print("\n  Sort your canonical table by whatever you filter on most --")
    print("  session, subject, time. It costs one sort at write time and it")
    print("  is what makes a query over a folder of files fast later.")


def demo_partition(ws):
    """
    Partitioning by split key.

    The directory layout is not cosmetic. If your honest evaluation holds out
    whole subjects or whole rooms, partitioning by that key means holding one
    out is a directory the reader never opens, rather than a boolean mask
    that could be wrong.
    """
    hdr("6. Partitioning by the split key",
        "Make an honest train/test split structural instead of remembered.")

    df = generate(120, 60, 50)
    root = ws / "canonical"
    pq.write_to_dataset(pa.Table.from_pandas(df, preserve_index=False),
                        root_path=str(root), partition_cols=["subject"],
                        compression="zstd")

    parts = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"  {root.name}/")
    for name in parts[:4]:
        files = list((root / name).glob("*.parquet"))
        print(f"    {name}/   {len(files)} file(s), "
              f"{size(root/name)/1024:.0f} KiB")
    print(f"    ... {len(parts) - 4} more\n" if len(parts) > 4 else "")

    # The partition column lives only in the folder names, which are text.
    # Read back with no further information, pyarrow has to guess its type,
    # and it guesses a dictionary type, which pandas turns into 'category'.
    # The int16 the contract declared is lost, exactly as with CSV.
    guessed = pq.read_table(root).to_pandas()["subject"].dtype
    print(f"  subject saved as int16, read back as: {guessed}")

    # The fix: declare the partition column's type when reading.
    part = ds.partitioning(pa.schema([("subject", pa.int16())]),
                           flavor="hive")
    declared = pq.read_table(root, partitioning=part).to_pandas()["subject"]
    print(f"  ...and with the type declared on read: {declared.dtype}\n")

    held_out = 3
    train = pq.read_table(root, partitioning=part,
                          filters=[("subject", "!=", held_out)])
    test = pq.read_table(root, partitioning=part,
                         filters=[("subject", "==", held_out)])

    tr = set(train.column("subject").to_pylist())
    te = set(test.column("subject").to_pylist())
    print(f"  holding out subject {held_out}:")
    print(f"    train {train.num_rows:>8,} rows, {len(tr)} subjects")
    print(f"    test  {test.num_rows:>8,} rows, {len(te)} subjects")
    print(f"    disjoint: {tr.isdisjoint(te)}")

    print("\n  The partition column is stored in the PATH, not in the file.")
    print("  It costs zero bytes, and a reader that filters on it skips")
    print("  whole directories without opening them.")
    print("\n  But the folder names carry no type, so declare the partition")
    print("  column's type when you read it back, or it changes silently.")
    print("\n  This is the same idea as generating both sides of a contract:")
    print("  the guarantee is built into the layout, not something anyone")
    print("  has to remember to check.")


def demo_append(ws):
    """
    Parquet is immutable, and the small-files problem that follows.

    You cannot append a row to a Parquet file. The footer is written last and
    describes everything before it. What you do instead is write many files
    and compact them -- and if you skip the compaction step, you get a
    thousand tiny files where metadata dominates.
    """
    hdr("7. You cannot append. Write many, then compact.",
        "The footer is written last, so a Parquet file is finished when closed.")

    df = generate(120, 60, 50)
    inc = ws / "incoming"
    inc.mkdir()

    # Simulate periodic rollups: many small writes, no compaction.
    n_parts = 120
    bounds = np.linspace(0, len(df), n_parts + 1, dtype=int)
    t0 = time.perf_counter()
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        df.iloc[a:b].to_parquet(inc / f"part-{i:04d}.parquet", index=False,
                                compression="zstd")
    t_many = time.perf_counter() - t0

    t0 = time.perf_counter()
    many = pq.read_table(inc)
    t_read_many = time.perf_counter() - t0

    one = ws / "compacted.parquet"
    t0 = time.perf_counter()
    pq.write_table(many, one, compression="zstd")
    t_compact = time.perf_counter() - t0

    t0 = time.perf_counter()
    pq.read_table(one)
    t_read_one = time.perf_counter() - t0

    print(f"  {n_parts} small files : {size(inc)/M:6.2f} MiB   "
          f"write {t_many*1000:5.0f} ms   read {t_read_many*1000:5.0f} ms")
    print(f"  1 compacted file : {size(one)/M:6.2f} MiB   "
          f"compact {t_compact*1000:5.0f} ms   read {t_read_one*1000:5.0f} ms")
    print(f"\n  overhead of not compacting: "
          f"{size(inc)/size(one):.2f}x the bytes, "
          f"{t_read_many/t_read_one:.1f}x the read time")

    print("\n  Each file carries its own footer, dictionary pages and row")
    print("  group metadata. Split the same data finely enough and that")
    print("  fixed cost dominates, and the dictionaries stop being shared.")
    print("\n  The practical shape: buffer live data somewhere mutable")
    print("  (SQLite, a log file), roll up to Parquet on a schedule, and")
    print("  compact the rollups periodically.")


def demo_pickle(ws):
    """
    Why 'it round-trips dtypes correctly' is not enough.

    Pickle scored 'exact' on schema fidelity in the benchmark. It is still
    the wrong choice for a canonical table, for a reason that has nothing to
    do with dtypes or speed.
    """
    hdr("8. The pickle problem",
        "Pickle preserves your dtypes perfectly. That is not the objection.")

    class Payload:
        def __reduce__(self):
            return (subprocess.run,
                    (["echo", "  >> arbitrary code ran while loading"],))

    p = ws / "innocent_looking.pkl"
    with open(p, "wb") as f:
        pickle.dump(Payload(), f)

    print(f"  {p.name} is {size(p)} bytes. It looks like data.")
    print(f"  Loading it with pandas.read_pickle:\n")
    # The payload's echo writes straight to the terminal, outside Python's
    # buffer. Flush first, or with output piped to a file the echo line
    # lands at the top of the file instead of here.
    sys.stdout.flush()
    try:
        pd.read_pickle(p)
    except Exception as e:                                # pragma: no cover
        print(f"  (blocked: {type(e).__name__})")

    print("\n  Unpickling is not parsing. It is executing a small program")
    print("  that reconstructs an object, and that program can do anything")
    print("  the process can do.")
    print("\n  Two further problems, less dramatic and more likely to bite:")
    print("    - a pickle written by one pandas or numpy version may not")
    print("      load under another")
    print("    - there is no schema to inspect without loading it, so you")
    print("      cannot audit a file before trusting it")
    print("\n  For data that never leaves your process, none of this")
    print("  matters. For a canonical table that outlives the code that")
    print("  wrote it, all of it does.")


DEMOS = {
    "write":     demo_write,
    "schema":    demo_schema,
    "internals": demo_internals,
    "rowgroups": demo_rowgroups,
    "stats":     demo_stats,
    "partition": demo_partition,
    "append":    demo_append,
    "pickle":    demo_pickle,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*", help=f"one or more of: {', '.join(DEMOS)}")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete the scratch directory")
    args = ap.parse_args()

    if args.list:
        for name, fn in DEMOS.items():
            first = [l.strip() for l in (fn.__doc__ or "").splitlines()
                     if l.strip()][0]
            print(f"  {name:<12}{first}")
        return

    unknown = [n for n in args.only if n not in DEMOS]
    if unknown:
        ap.error(f"unknown demo(s) {unknown}; choose from {list(DEMOS)}")
    chosen = args.only or list(DEMOS)

    print(f"  pyarrow {pa.__version__}   pandas {pd.__version__}")
    ws = Path(tempfile.mkdtemp(prefix="pq-howto-"))
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
