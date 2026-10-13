# 03-duckdb: querying and checking the dataset

Companion code for **Part 3** of the Edge AI Lab Notes series.

There are three scripts:

- `bench_duckdb.py` compares pandas and DuckDB on eight dataset tasks, to
  measure whether a query engine is worth adding.
- `duckdb_howto.py` runs six short demos of what a query engine makes
  possible.
- `validate_dataset.py` holds dataset checks with a command line that exits
  with an error when a check fails, so CI can use it directly.

```bash
pip install duckdb

python duckdb_howto.py                  # all six demos
python bench_duckdb.py                  # quick benchmark, 600,000 rows
python validate_dataset.py check DATA   # run the checks on your own data
```

## Why a query engine

[Part 2](https://bittobyteacademy.com/blog/why-parquet-for-edge-ai-data-measured-on-a-raspberry-pi/)
ended with the dataset stored in Parquet, with its data types intact. Part 3
is about what happens next.

pandas is not the problem. For one dataset that fits comfortably in memory,
pandas is fine, and an extra library may not be worth it. A query engine
becomes useful at three particular moments:

- when the dataset becomes **many files**, or a set of folders split by
  subject
- when a **second dataset** arrives and has to be compared with the first
- when **dataset checks should run in CI**, instead of in a notebook someone
  has to remember to open

This continues the idea from Part 1. Generating both sides of the data
contract made shared constants impossible to get out of step. Writing
dataset checks as queries does the same for facts about the data: they stop
being things someone checked once and become checks that run on every
change.

## The checks

```bash
python validate_dataset.py check DATA         # dataset checks
python validate_dataset.py split TRAIN TEST   # train and test overlap
```

`DATA`, `TRAIN` and `TEST` can each be a Parquet file, a folder of Parquet
files (including folders split by subject), or a pattern such as
`'rollups/*.parquet'`. Both commands exit with status 1 when something is
wrong, which is what makes a CI job fail.

`check` runs six checks, each a short SQL query:

| Check | What it catches |
|---|---|
| No subject is missing entirely | A subject dropped from the dataset |
| Every subject has at least one positive example | A subject whose labels were lost or never recorded |
| No channel is constant across the whole table | A sensor that never worked |
| No session has a constant channel | A sensor that stopped working during one recording |
| Class balance is within 20 to 1 for every subject | A subject so unbalanced that its results mean little |
| No missing values in any channel | Gaps in the data |

`split` counts the subjects and the recording sessions that appear in both
the training data and the test data. Both counts must be zero for an honest
evaluation.

Run `split` on **the files your training and evaluation code actually
read**. Checking two opposite filters of the same table proves nothing,
because they cannot overlap by definition. The first version of this check
made exactly that mistake.

To adapt the checks, edit `EXPECTED_SUBJECTS`, `CHANNELS` and the `CHECKS`
list at the top of `validate_dataset.py`.

## The demos

```bash
python duckdb_howto.py --list    # what each one shows
python duckdb_howto.py leakage   # run just one
```

| Demo | What it shows |
|---|---|
| `query` | Querying a Parquet file where it is, with no import step |
| `validate` | The checks passing on clean data and failing on data damaged on purpose |
| `leakage` | The split check passing on an honest split and failing on a random one |
| `union` | Two datasets with different columns queried as one table, revealing a units mismatch |
| `memory` | A hard memory limit: spilling to disk, then a clean failure |
| `arrow` | Passing query results to numpy without copying them |

Some notes on what they show:

**`validate` breaks the data on purpose.** It sets one subject's labels to
zero and flattens one session's gyroscope channel. Three checks fail, not
two, because the class balance check also fails as a result of the zeroed
labels. Checks often fail in related groups, so read the whole list before
deciding you have several separate problems. As with the sabotage script in
`01-contract`, a set of checks that has never failed tells you nothing.

**`leakage` compares two real splits.** Holding out subjects 3 and 7 passes.
Shuffling the rows and cutting them 80/20 fails, with all 12 subjects and
all 60 sessions on both sides. That is how a model scores far higher in
testing than it does in real use.

**`union` finds a units mismatch.** The second site reports acceleration in
m/s² instead of g, so its average acceleration is about nine times larger.
One query grouped by source catches it before the two datasets are trained
on together.

**`memory` needed three attempts to show anything.** An aggregation over
twelve groups needs almost no memory however large the input is, so it
passes any limit, and a sort that keeps only the first few rows is turned
into a much cheaper operation. Only a full sort written out to a file
genuinely has to hold all of the data. DuckDB also deletes its temporary
spill files as soon as it is done with them, so the folder is always empty
afterwards; the time taken is what shows that spilling happened. Below some
limit, DuckDB raises an exception the program can catch, instead of growing
until Linux's out-of-memory killer ends the process.

## The benchmark

```bash
python bench_duckdb.py --rows 2400000 --repeat 5 \
    --json ../results/my-duckdb.json

python bench_duckdb.py --rows 1200000 --repeat 5 --scaling \
    --json ../results/my-duckdb-scaling.json
```

Each task runs twice, once with pandas and once with DuckDB, on the same
Parquet files, in a newly started process with the file cache cleared, and
the median of the repeats is reported.

| Task | What it does |
|---|---|
| `full_aggregate` | Mean and standard deviation of three channels for each subject |
| `one_column` | The average of one column out of ten |
| `filtered_split` | Count the rows of every subject except the held-out one |
| `filtered_naive` | The same count without `columns=`, kept on purpose to show the cost |
| `hive_partition` | The same count, on data split into one folder per subject |
| `many_files` | Positive labels per subject, across 24 separate files |
| `validation` | Rows, positive examples, sessions and value ranges for each subject |
| `to_arrow` | Read all six sensor channels for every row, ready for a model |

Other options: `--only` to run some tasks, `--memory-limit` to give DuckDB a
memory limit, `--tmpdir`, and `--list` to describe the tasks.

### Fair to pandas

Every pandas task is given `columns=` for the columns it needs and
`filters=` where a condition applies, because that is the right way to use
pandas.

That rule changed the results. The first version of this benchmark left out
`columns=` on four tasks and reported DuckDB winning by up to 123 times. Most
of that came from pandas loading ten columns to count one. On the Pi, the
`filtered_naive` task takes 481 ms where the careful version takes 21 ms, for
the same answer and entirely within pandas.

### The agreement check

Both engines must return the same answer for every task, because a faster
wrong answer is not a result. The comparison uses a small tolerance, for the
same reason the golden vectors in `01-contract` do: two correct programs that
add up a column in a different order do not produce exactly the same
floating point result.

The check needs both a relative and an absolute tolerance. On one run, the
mean of a channel came out at `-0.0022`, and the two engines differed by
`3.6e-9`. That is tiny in absolute terms but `1.6e-6` relative to such a
small number, so a relative tolerance alone would have reported a false
failure.

### The memory sweep

`--scaling` runs `full_aggregate` at a quarter, half, one and two times the
`--rows` value, so `--rows 1200000` gives 300,000 to 2.4 million rows. It
runs each size half as many times as `--repeat`, which is enough because
memory figures barely vary between runs. It shows how memory grows with the
dataset for each engine.

### Results on a Raspberry Pi 5

From `../results/pi5-duckdb.json` and `../results/pi5-duckdb-scaling.json`:

- DuckDB was between **1.7 and 8.2 times faster**, leaving aside the
  deliberately unfair `filtered_naive`. pandas won `to_arrow`, 332 ms against
  429 ms.
- DuckDB needed between 5.5 and 14.3 times less **working memory** (peak
  memory minus the memory in use before the task), but only 1.12 to 1.96
  times less **peak memory**, because loading DuckDB itself costs about
  33 MiB.
- pandas' working memory grew by about 68 MiB per million rows, and
  DuckDB's by about 3 MiB.
- At 300,000 rows, pandas used less memory in total than DuckDB. The two
  cross somewhere between 300,000 and 600,000 rows on this hardware.

## When not to bother

- **One dataset that fits comfortably in memory.** pandas is fine. Add
  `columns=` to your reads, and you have most of the speed for free.
- **Loading all of the data with no filtering.** That is the one task pandas
  won.
- **Looking up single rows, or holding live device state.** That is a job
  for SQLite.
- **On the microcontroller.** None of this runs there. This is code for a
  gateway or a workstation.

## Licence

Code in this folder is under the MIT licence. This README is under CC BY 4.0. See the
[top-level README](../README.md#licence) for details.
