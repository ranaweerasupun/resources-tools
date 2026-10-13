# edge-ai-toolkit

[![edge-ai-toolkit](https://github.com/ranaweerasupun/resources-tools/actions/workflows/edge-ai-toolkit.yml/badge.svg)](https://github.com/ranaweerasupun/resources-tools/actions/workflows/edge-ai-toolkit.yml)

Companion code for the **[Edge AI Lab Notes](https://bittobyteacademy.com/series/edge-ai-lab-notes/)**
series on Bit to Byte Academy.

Everything here runs on an ordinary laptop or a Raspberry Pi. You do not need
a sensor board, a downloaded dataset or any vendor tools: only Python, and a
C compiler for Part 1. Every example uses a synthetic data generator, so
anyone can reproduce the results.

## What is here

```
edge-ai-toolkit/
├── synth/           the synthetic sensor data generator, used by every part
├── common/          shared measurement helpers for the benchmarks
├── 01-contract/     Part 1: data contracts, golden vectors, sabotage tests
├── 02-formats/      Part 2: ten storage options benchmarked, Parquet in practice
├── 03-duckdb/       Part 3: pandas against DuckDB, dataset checks for CI
└── results/         benchmark results from a Raspberry Pi 5
```

| Part | Folder | What it shows |
|---|---|---|
| 1. Data contracts | `01-contract/` | Keeping Python training code and C firmware in agreement, with checks that fail the build when they drift |
| 2. Why Parquet | `02-formats/` | Ten storage options compared on size, speed, memory and whether data types survive |
| 3. Parquet with DuckDB | `03-duckdb/` | A fair pandas and DuckDB comparison, and dataset checks that CI can fail on |
| 4. Parquet and Arrow | coming | Storage on disk against data in memory, and when each fits |

Each folder has its own README with the details, and each part of the series
explains the ideas behind the code.

## Setup

The results in the series were measured with Python 3.11.

```bash
git clone https://github.com/ranaweerasupun/resources-tools.git
cd resources-tools/edge-ai-toolkit
pip install -r requirements.txt
```

`requirements.txt` pins the exact library versions used for every result in
the series. `h5py` and `duckdb` are optional. The format benchmark skips any format whose
library is missing instead of failing, and only Part 3 needs `duckdb`.

Part 1 also needs a C compiler (`cc` or `gcc`) and `make`. Both come with
Raspberry Pi OS, most Linux distributions, and the macOS command line tools.

## Quick start

```bash
# Part 1: run every check, then break the C code on purpose
cd 01-contract && make check && python sabotage.py && cd ..

# Part 2: a quick version of the format benchmark, and the Parquet demos
cd 02-formats && python bench_formats.py && python parquet_howto.py && cd ..

# Part 3: a quick version of the DuckDB benchmark, and the demos
cd 03-duckdb && python bench_duckdb.py && python duckdb_howto.py && cd ..
```

The quick runs take a minute or two on a laptop. The full runs used in the
series are listed in each folder's README, and take longer, especially on a
Raspberry Pi.

## Continuous integration

Every change to this folder runs two jobs on GitHub Actions, defined in
`.github/workflows/edge-ai-toolkit.yml` at the root of the repository:

- **Part 1** runs `make check` and `python sabotage.py`, so a change that
  lets the Python and C sides drift apart, or that stops the golden test
  from catching a known mistake, fails the build. It then confirms that the
  sabotage script restored every file.
- **Parts 2 and 3** run every demo and a small version of each benchmark,
  to confirm the code still works with the pinned libraries. Both
  benchmarks exit with an error if a format fails, or if pandas and DuckDB
  return different answers.

Timings from a CI runner are not meaningful, so the benchmark results there
are not kept. The published numbers all come from the Raspberry Pi runs in
`results/`.

## The synthetic data

```bash
python synth/synth.py --sessions 200 --seconds 120
```

The generator produces a table with one row per sensor reading: six motion
sensor channels plus columns for time, recording session, subject and label.

The data is designed to behave like real sensor data, because the benchmark
results depend on it. Random numbers from `np.random.randn()` have no
patterns, so no file format can compress them, and a benchmark built on them
would wrongly conclude that Parquet's encoding does not help. Real sensor
data differs in three ways, and the generator reproduces all three:

- The sensor reports whole-number readings from a 16-bit converter, so each
  channel can only take a limited set of values.
- Consecutive readings are close to each other, because real signals change
  smoothly.
- The session, subject and label columns repeat for thousands of rows at a
  time.

## Results

The `results/` folder holds the output of every benchmark and demo run on a
Raspberry Pi 5 with 8 GB of memory, running Raspberry Pi OS Bookworm from an
SD card. These are the numbers quoted in the series.

| File | Produced by |
|---|---|
| `pi5-01-check.txt` | `make check` in `01-contract` |
| `pi5-01-sabotage.txt` | `python sabotage.py` in `01-contract` |
| `pi5-formats-float32.json` | the full format benchmark |
| `pi5-formats-int16.json` | the same benchmark with `--raw-ticks` |
| `pi5-formats-float32-nofsync.json` | the same benchmark with `--no-fsync`, wrong on purpose |
| `pi5-parquet-demos.txt` | `python parquet_howto.py` |
| `pi5-duckdb.json` | the full DuckDB benchmark |
| `pi5-duckdb-scaling.json` | the DuckDB memory sweep |
| `pi5-duckdb-demos.txt` | `python duckdb_howto.py` |

Each JSON file records the hardware, operating system, storage device,
library versions and processor temperature of its run, so results from
different machines can be compared fairly. Results from other boards,
especially smaller ones, are very welcome.

## Licence

- **Code** is under the MIT licence, in [`LICENSE`](../LICENSE) at the root of
  this repository.
- **The benchmark results** in `results/` and **the documentation**, meaning
  the READMEs, are under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You may reuse them
  for any purpose, including commercially, as long as you credit
  Supun Sriyananda and link to this repository.

## Citing this work

If you use this code or these results in research, please cite the
repository. GitHub's **Cite this repository** button, in the sidebar, gives a
ready-made reference from [`CITATION.cff`](../CITATION.cff).
