# 02 — Why Parquet for edge AI data

Companion code for **post 2** of the Edge AI series.

Two scripts. `bench_formats.py` answers *which format*. `parquet_howto.py`
answers *now what do I type*.

```bash
python parquet_howto.py            # eight practical demos
python parquet_howto.py --list     # what each one shows
python parquet_howto.py partition  # run just one
```

| Demo | Shows |
|---|---|
| `write` | declare dtypes, then verify the round-trip instead of assuming |
| `schema` | read the schema and statistics without touching the data |
| `internals` | what encoding does before any compression codec runs |
| `rowgroups` | row group sizing, and reading one column out of ten |
| `stats` | sorted data can be skipped; shuffled data cannot |
| `partition` | partitioning by split key, so an honest split is structural |
| `append` | Parquet is immutable — write many files, then compact |
| `pickle` | why "it preserves dtypes" is not enough |

---

## The benchmark

```bash
python bench_formats.py                       # quick check, ~600k rows
python bench_formats.py --sessions 400 --repeat 5 \
    --json ../results/myhost-float32.json     # the real run, 2.4M rows
python bench_formats.py --sessions 400 --repeat 5 --raw-ticks \
    --json ../results/myhost-int16.json       # int16 ticks instead of float32
```

Start with the quick run to confirm the environment works before committing
to the long one.

---

## What it measures

Ten formats, on the same canonical sensor table:

| | |
|---|---|
| **size** | on disk, after compression |
| **write** | to durability, fsync included |
| **read** | with the page cache dropped first |
| **peak RAM** | in a fresh process doing only that read |
| **fidelity** | does the round-trip return the schema you wrote? |

Formats: `csv`, `csv.gz`, `json`, `pickle`, `npz`, `hdf5`, `parquet-none`,
`parquet-snappy`, `parquet-zstd`, `feather-zstd`.

Useful flags: `--only`, `--skip`, `--sessions`, `--repeat`, `--raw-ticks`,
`--tmpdir`, `--json`.

---

## Method notes

These are the things that make benchmark numbers wrong. Worth repeating
wherever the results get quoted.

**Fresh subprocess per read.** Peak RSS is a high-water mark for the life of
a process, so measuring several formats in one process reports the largest
one for all of them.

**`VmHWM`, not `getrusage`.** Linux keeps `ru_maxrss` in the signal struct,
and that struct survives `execve`. A child launched from a parent that has
touched 2 GiB reports 2 GiB as its own peak before running a line of its own
code. Measured: parent peak 1212 MiB → child `ru_maxrss` 1212 MiB, child
`VmHWM` 11 MiB. `VmHWM` lives in the mm_struct, which `execve` replaces.

**Page cache dropped per file** with `posix_fadvise(DONTNEED)`. Without it,
the second read of a file is served from RAM and you are timing memory
bandwidth. Needs no root.

**Writes are fsync'd, data and directory.** Timing only as far as the page
cache is not timing to the card. It made pickle appear to write 88 MB in
0.043 s — over 2 GB/s, faster than any SD card physically accepts — and it
made zstd look 81% more expensive than it actually is.

**Median of N runs.** A single timing on an SD card is noise.

**Check the thermal line.** On a Pi, a throttled run produces slow numbers
that have nothing to do with the formats being compared. The harness prints
`get_throttled` decoded into plain English before and after, and warns if
throttling occurred.

---

## Reading the output

The environment block is part of the result, not preamble. A format
benchmark that does not state the storage medium is not reproducible — an SD
card and an NVMe SSD differ by more than an order of magnitude on write
throughput.

```
  device     Raspberry Pi 5 Model B Rev 1.1
  cpu        aarch64, 4 cores
  memory     7.9 GiB
  storage    /dev/mmcblk0p2 [SD/eMMC SATC]
  thermal    57.1'C  clean
  packages   numpy 2.4.6 (66.6 MiB)  pandas 3.0.5 (71.5 MiB)
             pyarrow 25.0.1 (140.7 MiB)  duckdb 1.5.5 (53.0 MiB)
```

Installed package size is measured from distribution metadata rather than the
package directory, because several libraries ship their heavy shared object
outside the importable package — walking `Path(module.__file__).parent`
reports duckdb at 0.9 MiB instead of 53.0 MiB.

That number belongs in any honest write-up. Parquet and DuckDB together add
roughly 190 MiB to a device image, which is a real cost nobody in the data
engineering world ever mentions.

---

## The `--raw-ticks` variant

Stores int16 ADC ticks instead of scaled float32. The comparison is more
interesting than "store ticks, save space":

Every format rewards you for storing ticks — 20 to 42% — **except Parquet**,
where the difference is under 1%. Parquet has no 16-bit physical type, so an
int16 column becomes `INT32` with a logical annotation. Both int16 and
float32 are then 4-byte physical types that dictionary-encode over the same
set of distinct values, producing byte-identical index streams.

The saving does not disappear though, it moves: choosing int16 cuts the
in-memory dataframe by a third and peak read RAM by about 20%. On Parquet the
dtype decision is a memory decision, not a disk decision.

Run both variants and compare the two JSON files.
