"""
validate_dataset.py -- dataset checks that a CI job can fail on.

Two commands:

    python validate_dataset.py check DATA
        Run every check in CHECKS against DATA (a Parquet file, a folder of
        Parquet files, or a glob). Exits with status 1 if any check fails.

    python validate_dataset.py split TRAIN TEST
        Check that the training data and the test data share no subjects and
        no sessions. Exits with status 1 if they overlap.

An exit status of 1 is what makes a CI job fail. Printing a warning is not
enough: nobody reads the log of a job that passed.

The checks are plain SQL, run by DuckDB directly against the files, so there
is no load step and nothing to keep in sync with a DataFrame somewhere else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

EXPECTED_SUBJECTS = 12
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")

# Each check is (description, SQL returning one number, test the number must
# pass). The SQL reads from a view called t, created by run_checks().
CHECKS = [
    ("no subject is missing entirely",
     "SELECT count(DISTINCT subject) FROM t",
     lambda v: v == EXPECTED_SUBJECTS),

    ("every subject has at least one positive example",
     """SELECT count(*) FROM (
            SELECT subject FROM t GROUP BY subject HAVING sum(label) = 0)""",
     lambda v: v == 0),

    ("no channel is constant across the whole table",
     "SELECT count(*) FROM (SELECT 1 FROM t HAVING "
     + " OR ".join(f"min({c}) = max({c})" for c in CHANNELS) + ")",
     lambda v: v == 0),

    ("no session has a constant channel",
     "SELECT count(*) FROM (SELECT session FROM t GROUP BY session HAVING "
     + " OR ".join(f"min({c}) = max({c})" for c in CHANNELS) + ")",
     lambda v: v == 0),

    # A subject with no positive examples is counted as having one, so the
    # ratio stays finite. It then equals the subject's row count, which is
    # far above 20, so this check fails too. That is expected: see the note
    # about correlated failures in the series.
    ("class balance is within 20:1",
     """SELECT max(1.0 * n / p) FROM (
            SELECT count(*) AS n, greatest(sum(label), 1) AS p
            FROM t GROUP BY subject)""",
     lambda v: v <= 20),

    ("no missing values in any channel",
     "SELECT count(*) FROM t WHERE "
     + " OR ".join(f"{c} IS NULL" for c in CHANNELS),
     lambda v: v == 0),
]


def _source(path: str) -> str:
    """A DuckDB read_parquet() expression for a file, folder or glob."""
    p = Path(path)
    if p.is_dir():
        return (f"read_parquet('{p}/**/*.parquet', "
                f"hive_partitioning = true, union_by_name = true)")
    return f"read_parquet('{path}')"


def run_checks(path: str, con=None):
    """Run CHECKS against the data at path. Returns [(name, value, ok)]."""
    con = con or duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW t AS SELECT * FROM {_source(path)}")
    results = []
    for name, sql, test in CHECKS:
        value = con.execute(sql).fetchone()[0]
        results.append((name, value, bool(test(value))))
    return results


def check_split(train: str, test: str, con=None):
    """
    Count the subjects and sessions that appear in both the training data
    and the test data. Both counts must be zero for an honest evaluation.

    The two inputs must be the datasets that training and evaluation really
    read. Checking two filters of the same table proves nothing, because
    complementary filters cannot overlap by definition.
    """
    con = con or duckdb.connect()
    out = {}
    for key in ("subject", "session"):
        out[key] = con.execute(f"""
            SELECT count(*) FROM
              (SELECT DISTINCT {key} FROM {_source(train)}) AS a
              JOIN
              (SELECT DISTINCT {key} FROM {_source(test)}) AS b
              USING ({key})
        """).fetchone()[0]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="run the dataset checks")
    c.add_argument("data")
    s = sub.add_parser("split", help="check a train/test split for overlap")
    s.add_argument("train")
    s.add_argument("test")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        results = run_checks(args.data)
        for name, value, ok in results:
            print(f"  {'ok  ' if ok else 'FAIL'}  {name:<48} = {value}")
        failed = sum(not ok for _, _, ok in results)
        print(f"\n  {len(results) - failed}/{len(results)} passed")
        return 1 if failed else 0

    overlap = check_split(args.train, args.test)
    for key, n in overlap.items():
        print(f"  {key}s in both train and test: {n}")
    leaked = any(overlap.values())
    print("  LEAKAGE" if leaked else "  disjoint")
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
