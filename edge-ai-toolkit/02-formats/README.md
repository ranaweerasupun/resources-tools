# 02-formats: choosing a file format for edge AI data

Companion code for **Part 2** of the Edge AI Lab Notes series.

There are two scripts. `bench_formats.py` measures ten storage options to
answer *which format to use*. `parquet_howto.py` runs eight short demos that
show *how to use Parquet well*.

```bash
python parquet_howto.py            # all eight demos
python parquet_howto.py --list     # what each one shows
python parquet_howto.py partition  # run just one
```

| Demo | What it shows |
|---|---|
| `write` | Declare data types explicitly, then check they survived saving and loading |
| `schema` | Read a file's columns, types and statistics without reading its data |
| `internals` | How Parquet encodes each column before any compression runs |
| `rowgroups` | Row group sizes, and reading one column out of ten |
| `stats` | Why sorted data lets a reader skip blocks and shuffled data does not |
| `partition` | Splitting a dataset into folders by subject, and declaring the folder column's type when reading it back |
| `append` | Parquet files cannot be appended to, so write many and combine them |
| `pickle` | Why pickle is unsafe for shared data, even though it keeps every type |

The demos do not clear the operating system's file cache between reads, so
their timings are rough illustrations that vary from run to run. Use the
benchmark for measurements.

## The benchmark

```bash
python bench_formats.py                     # quick check, about 600,000 rows

python bench_formats.py --sessions 400 --repeat 5 \
    --json ../results/my-float32.json       # the full run, 2.4 million rows

python bench_formats.py --sessions 400 --repeat 5 --raw-ticks \
    --json ../results/my-int16.json         # raw 16-bit readings instead of float32
```

Start with the quick run to check that everything works before the full one,
which takes a while on a Raspberry Pi.

### What it measures

Every option stores the same table, and each is measured on five things:

| Measure | How |
|---|---|
| **Size** | The file size on disk |
| **Write time** | Until the data is safely stored, including `fsync` |
| **Read time** | With the operating system's cached copy of the file cleared first |
| **Peak memory** | In a newly started process that does only that read |
| **Types preserved** | Whether every column comes back with the type it was saved with |

The ten options are `csv`, `csv.gz`, `json` (one JSON record per line),
`pickle`, `npz`, `hdf5`, Parquet with no compression (`parquet-none`),
Parquet with snappy and with zstd compression (`parquet-snappy`,
`parquet-zstd`), and Feather with zstd (`feather-zstd`).

Useful options: `--only` and `--skip` to choose formats, `--sessions`,
`--seconds` and `--rate-hz` to change the dataset size, `--repeat`,
`--raw-ticks`, `--tmpdir` to choose where test files are written, `--json`
to save the results, and `--no-fsync` (see below).

### Results on a Raspberry Pi 5

From `../results/pi5-formats-float32.json`: 2.4 million rows, 84.7 MiB in
memory, stored on the Pi's SD card.

| Format | Size | Write | Read | Peak memory | Types preserved |
|---|---|---|---|---|---|
| parquet-zstd | 27.0 MiB | 1.40 s | 0.44 s | 493 MiB | Yes |
| npz | 29.8 MiB | 8.01 s | 0.54 s | 350 MiB | Yes |
| feather-zstd | 30.1 MiB | 1.27 s | 0.38 s | 470 MiB | Yes |
| hdf5 | 30.5 MiB | 3.17 s | 0.57 s | 373 MiB | Yes |
| parquet-snappy | 31.2 MiB | 1.40 s | 0.47 s | 449 MiB | Yes |
| parquet-none | 31.4 MiB | 1.42 s | 0.46 s | 458 MiB | Yes |
| csv.gz | 48.7 MiB | 44.48 s | 3.46 s | 672 MiB | No |
| pickle | 84.7 MiB | 3.05 s | 0.94 s | 264 MiB | Yes |
| csv | 176.1 MiB | 31.89 s | 3.31 s | 678 MiB | No |
| json | 371.4 MiB | 14.59 s | 17.67 s | 4,631 MiB | No |

File sizes are exact and repeat on every run. Times vary somewhat,
especially write times on an SD card. Parquet's peak memory varied between
about 440 and 500 MiB across runs, so the three Parquet options should not be
ranked against each other on memory.

## How the measurements are kept honest

These are the details that most often make format benchmarks wrong. They
are worth stating whenever the numbers are quoted.

**Each read runs in a new process.** A process's peak memory is a high-water
mark that never goes down, so measuring several formats in one process would
report the largest peak for all of them.

**Peak memory comes from `VmHWM`, not `getrusage`.** On Linux, a new process
inherits the parent's `ru_maxrss` value and keeps it. A child started by a
parent that once used 1.2 GB reports 1.2 GB before doing anything. Measured
directly: parent peak 1,211 MiB, child `ru_maxrss` 1,211 MiB, child `VmHWM`
11 MiB. `VmHWM` is reset when a new program starts, so it gives the right
answer.

**The cached copy of each file is cleared before reading it**, using
`posix_fadvise` with `DONTNEED`. Otherwise the second read of a file comes
from memory, and you measure memory speed instead of the format. This does
not need administrator rights.

**Writes are timed until the data is stored**, with `fsync` on both the file
and its folder. Without it, the timer stops when the operating system has
merely accepted the data into memory.

**Each measurement is repeated and the median reported**, because a single
timing on an SD card is mostly noise.

**The processor temperature is recorded.** On a Raspberry Pi, a run that
overheats and slows itself down (thermal throttling) produces slow numbers
that have nothing to do with the formats. The benchmark reports the
temperature and throttling state before and after, and warns if throttling
happened.

### Reproducing the fsync mistake

`--no-fsync` times writes without `fsync`, repeating on purpose the mistake
in the first version of this benchmark. It prints a warning, because the
write times it produces are wrong by design.

On the Pi (`../results/pi5-formats-float32-nofsync.json`), it made pickle
appear to write 88.8 MB in 0.04 seconds, about 2 GB per second, which no SD
card can do. It also made the three Parquet options about 50% *slower*,
probably because unfinished writes from the formats measured just before
were still being written to the card. The errors go in both directions,
which is the point: without `fsync`, the results are unpredictable.

## The environment report

Every run reports what it ran on, and the JSON results include it. The
storage device matters most: an SD card and a fast SSD differ by more than
ten times in write speed, so a format benchmark that does not state its
storage cannot be reproduced.

```
  device     Raspberry Pi 5 Model B Rev 1.0
  cpu        aarch64, 4 cores
  memory     7.9 GiB
  os         Debian GNU/Linux 12 (bookworm)
  storage    /dev/mmcblk0p2 [SD/eMMC SN32G]
  packages   numpy 2.4.6 (66.6 MiB), pandas 3.0.6 (71.3 MiB),
             pyarrow 26.0.0 (150.8 MiB), duckdb 1.5.6 (53.1 MiB)
```

Installed library sizes are read from each package's list of installed
files. Adding up the files in a package's own folder gives the wrong answer
for some libraries, because they keep their main compiled file elsewhere:
that method reports duckdb at 0.9 MiB instead of 53 MiB.

This number matters on small devices. pyarrow and duckdb together add about
204 MiB to a device's software image.

## Storing raw 16-bit readings

`--raw-ticks` stores the sensor's raw 16-bit readings (`int16`) instead of
values converted to real units (`float32`). Compare the two JSON files from
the Pi (`pi5-formats-float32.json` and `pi5-formats-int16.json`):

- Every format except Parquet gets smaller, by between 21% and 42%.
- Parquet changes by 1% at most. The two uncompressed Parquet files differ
  by 18 bytes.

Parquet has no 16-bit storage type, so an `int16` column is stored as a
32-bit integer with a note to convert it back. Both versions therefore use
4-byte values, and both are dictionary-encoded over the same set of
distinct readings.

The saving moves into memory instead: with `int16` columns, the table in
pandas shrinks from 84.7 MiB to 57.2 MiB, and the peak memory needed to read
the zstd Parquet file fell by about 17% in these runs.
