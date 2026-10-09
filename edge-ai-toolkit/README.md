# edge-ai-toolkit

Runnable companion code for the **Edge AI** series.

Everything here runs on a laptop or a Raspberry Pi. No board, no sensor, no
downloaded dataset, no vendor SDK — just Python and a C compiler. Every
example is built on a synthetic sensor generator so the results are
reproducible by anyone.

---

## Layout

```
edge-ai-toolkit/
├── synth/           synthetic sensor data, shared by every post
├── 01-contract/     post 1 — data contracts, golden vectors, sabotage tests
├── 02-formats/      post 2 — why Parquet, measured against 9 other formats
└── results/         benchmark output from real hardware
```

| Post | Directory | What it demonstrates |
|---|---|---|
| 1 — Data contracts | `01-contract/` | Making Python↔C disagreement structurally impossible |
| 2 — Why Parquet | `02-formats/` | Format comparison on size, speed, memory, schema fidelity |
| 3 — Parquet + DuckDB | `03-duckdb/` | Validation as a query instead of a script |
| 4 — Parquet vs Arrow | *coming* | On-disk against in-memory, and when each wins |

---

## Setup

```bash
git clone <this repo>
cd edge-ai-toolkit
pip install numpy pandas pyarrow h5py duckdb
```

`h5py` and `duckdb` are optional — the benchmark skips formats whose library
is missing rather than failing. A C compiler (`cc` or `gcc`) and `make` are
needed for post 1 only; both ship with Debian, Raspberry Pi OS, macOS command
line tools, and every Linux distribution.

---

## 01-contract — data contracts

```bash
cd 01-contract
make generate      # regenerate every generated file from the contract
make check         # the full three-level check, as CI would run it
python sabotage.py # break the C code ten ways, prove each is caught
```

An edge AI system trains a model in one language and runs it in another.
Failures at that seam do not crash — the model is fed something it never
trained on and returns a confident, well-formed, wrong answer.

Three levels of defence, each catching what the one below cannot:

| Level | What disagrees | Mechanism |
|---|---|---|
| 1 | a **constant** | one source, generated files, hash + `--check` |
| 2 | a **function** | golden vectors |
| 3 | a **decision nobody made** | `#error` until it is decided |

Level 2 is the part most people skip:

> A constant that disagrees is caught by a hash.
> A function that disagrees is not.

Ten sabotages, ten caught. See `01-contract/README.md`.

---

## 02-formats — why Parquet

```bash
cd 02-formats
python bench_formats.py                                    # quick, ~600k rows
python bench_formats.py --sessions 400 --repeat 5 \
    --json ../results/myhost-float32.json                  # the real run
python bench_formats.py --sessions 400 --repeat 5 --raw-ticks \
    --json ../results/myhost-int16.json                    # int16 variant
```

`parquet_howto.py` covers the practical side: writing with explicit dtypes,
inspecting a file without reading it, row groups, partitioning by split key,
the fact that Parquet cannot be appended to, and why pickle is the wrong
choice even though it preserves dtypes perfectly.

`bench_formats.py` measures ten formats — csv, csv.gz, json, pickle, npz, hdf5, three Parquet
codecs, and Arrow IPC — on file size, write time to durability, cold read
time, peak RAM, and whether the round-trip preserves your schema.

The harness reports its own environment: board model, core count, RAM,
storage medium, installed package sizes, and thermal state before and after.
Paste that block straight into a write-up.

Three things it does that most format benchmarks get wrong, and which are
worth stating whenever the numbers are quoted:

- **Each read runs in a fresh subprocess.** Peak RSS is a high-water mark for
  the process lifetime, so measuring several formats in one process reports
  the largest for all of them. It reads `VmHWM`, not `getrusage`, because
  `ru_maxrss` survives `exec` and a child inherits its parent's peak.
- **The page cache is dropped per file** with `posix_fadvise(DONTNEED)`
  before each read. Without it you are timing memory bandwidth. No root
  needed.
- **Writes are fsync'd.** Timing to the page cache is not timing to the card:
  it made pickle look like it wrote 88 MB in 0.04 s, and made zstd look 81%
  more expensive than it is.

---

## synth — the shared dataset

```bash
python synth/synth.py --sessions 200 --seconds 120
```

Quantised, autocorrelated channels with realistic repetitive metadata.

This matters more than it sounds. `np.random.randn()` produces float32 that
uses the full mantissa and compresses to nothing, so a benchmark built on it
measures Parquet at roughly 1.0x and concludes columnar storage does not
help. Real sensor data comes from a fixed-point ADC, is autocorrelated, and
carries session and label columns that repeat for thousands of rows. The
generator reproduces all three properties, so numbers measured with it
transfer to real data.

---

## results

Benchmark JSON from real hardware. Add your own — the environment block in
each file records exactly what it was measured on, so results from different
boards stay comparable.

---

## Licence

MIT. Take any of it.
