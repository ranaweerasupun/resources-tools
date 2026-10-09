# 03 — Parquet with DuckDB

Companion code for **post 3** of the Edge AI series.

Two scripts. `bench_duckdb.py` measures whether a query engine is worth the
dependency. `duckdb_howto.py` shows what to type.

```bash
pip install duckdb

python duckdb_howto.py                      # six demos
python bench_duckdb.py --rows 600000        # quick check
```

---

## The argument

Post 2 ended with a canonical table in Parquet with a schema that survives a
round trip. This post is about what you do with it once there is more than
one file and more than one dataset.

The claim is not that pandas is bad. For a single dataset that fits in
memory, pandas is fine and the extra dependency is not obviously worth it. A
query engine earns its place at three specific moments:

- when the canonical table becomes **many files**, or a partitioned tree
- when a **second data source** arrives and has to be compared with the first
- when **validation should run in CI** rather than in a notebook somebody
  remembers to open

The through-line from post 1: generating both sides of a contract made a
constant impossible to get wrong. Writing dataset checks as queries does the
same thing one layer up. Facts about your data stop being things somebody
verified once and become assertions that run.

---

## The benchmark

```bash
python bench_duckdb.py --rows 2400000 --repeat 5 \
    --json ../results/myhost-duckdb.json

python bench_duckdb.py --rows 600000 --repeat 4 --scaling \
    --json ../results/myhost-duckdb-scaling.json
```

Seven tasks, each run twice — once through pandas, once through DuckDB — over
the same Parquet files.

| Task | What it measures |
|---|---|
| `full_aggregate` | mean and std per subject; a full scan |
| `one_column` | one column out of ten (pandas gets `columns=`) |
| `filtered_split` | everything except one held-out subject |
| `filtered_naive` | the same count, written without `columns=` |
| `hive_partition` | the same query against a partitioned tree |
| `many_files` | scanning un-compacted rollup files |
| `validation` | a realistic dataset-check suite |
| `to_arrow` | file to something a model can consume |

The pandas side is written the way a careful person would write it. Every
task gives pandas `columns=` for the columns it needs and `filters=` where a
predicate applies, because both exist and both are the right way to use it.

**That fairness changed the headline.** My first version omitted `columns=`
on four tasks and reported DuckDB at up to 123x. Most of that gap was pandas
materialising ten columns to count one — 152 ms against 6 ms for the same
answer, *entirely within pandas*. Corrected, those tasks land between 1.2x
and 6x.

The `filtered_naive` task keeps the naive version on purpose, because the
cost of forgetting is worth knowing. But it is labelled as what it is rather
than passed off as an engine difference.

### The agreement check

Every task returns a value from both engines and the harness compares them. A
faster wrong answer is not a result, and this check earned its place
immediately: it caught two apparent mismatches on the first run.

Both turned out to be float summation order, about `1e-8` relative on a mean
over 300,000 float32 values, which is float32 noise. The fix was to the
*check*, not the code — for exactly the reason the golden vectors in
`01-contract` use tolerances. Two correct implementations that sum a column
in different orders will not produce identical floats.

The comparison needs **both** a relative and an absolute tolerance. Relative
alone fails whenever the true answer happens to sit near zero: on one run the
mean of a channel came out at `-0.0022`, the two engines differed by `3.6e-9`
absolute, and that is `1.6e-6` relative — a spurious failure caused entirely
by where the answer landed.

### The scaling sweep

```bash
python bench_duckdb.py --rows 600000 --scaling
```

Runs the same aggregation at four dataset sizes and reports how peak memory
grows for each engine. This is the honest way to answer "what happens when
the data does not fit in RAM" without generating a file larger than the
machine: if one column grows with the data and the other stays flat, the
point at which pandas stops fitting is arithmetic.

---

## The demos

```bash
python duckdb_howto.py --list
```

| Demo | Shows |
|---|---|
| `query` | querying Parquet in place, no import step |
| `validate` | dataset checks as assertions, with defects injected on purpose |
| `leakage` | proving a subject-wise split is disjoint, in SQL |
| `union` | two sources with different columns, via `union_by_name` |
| `memory` | a hard memory ceiling, spilling, and a clean failure |
| `arrow` | zero-copy from query result into numpy |

Two of these deserve a note.

**`validate` fails on purpose.** It injects two defects — a subject with no
positive examples, and a session with a flatlined channel — and three of six
checks trip. The third fails as a *consequence* of the first, which is worth
seeing: checks fail in correlated groups, so read the whole list before
concluding you have three separate problems. Same reasoning as the sabotage
script in `01-contract`: a suite that has never failed tells you nothing.

**`memory` shows a real boundary.** Getting this demo to mean anything took
three attempts. An aggregation over twelve groups needs almost no memory
however large the input, so it passes any budget. `ORDER BY ... LIMIT` is
rewritten into a top-N operator that never materialises the relation. What
works is sorting every row and writing the result out.

DuckDB also deletes its spill files the moment it is finished with them, so
measuring the temp directory afterwards always reads zero. The observable is
the time, and then the failure: below some budget it raises a catchable
exception rather than growing until the kernel's OOM killer takes the process
out. That difference is the whole argument on a gateway.

---

## When not to bother

- **One dataset that fits in memory.** pandas is fine, and duckdb is another
  ~53 MiB on the device image.
- **Row-at-a-time access or live state.** That is SQLite's job.
- **On the microcontroller.** None of this runs there. This is gateway and
  workstation code.
