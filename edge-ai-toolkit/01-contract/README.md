# 01-contract: data contracts for edge AI

Companion code for **Part 1** of the Edge AI Lab Notes series. Everything
here runs on a laptop with Python, numpy, a C compiler and `make`.

```bash
make generate        # regenerate every generated file from the contract
make run             # build and run the C golden test
make check           # run all three levels of checks, as CI would
python sabotage.py   # break the C code ten ways and confirm each is caught
```

## The problem

An edge AI system usually trains its model in Python and runs it on a device
programmed in C. The two programs must agree exactly on what a window of
sensor data is: how many readings it holds, the order of the channels, the
units, how a missing reading is marked, and where each feature goes in the
list the model receives.

When they disagree, nothing crashes. The model simply receives data it was
never trained on, and gives answers that look normal but are wrong.

Asking everyone to be careful does not prevent this, because it depends on
people remembering. This code takes a different approach: it sets up the
project so that the two sides cannot drift apart without a check failing.
There are three levels, and each catches a kind of mistake the previous one
cannot.

| Level | What can disagree | How it is caught | Files |
|---|---|---|---|
| 1 | A **constant**, such as the window length | One source file generates both languages; `--check` detects drift | `data_contract.py` |
| 2 | A **function**, such as the feature calculation | Golden vectors that both languages must reproduce | `gen_golden.py`, `test_golden.c`, `test_contract.py` |
| 3 | A **value nobody has decided yet** | The firmware refuses to compile until it is set | `contract_runtime.h` |

The series covers each level in its own post:
[1.1](https://bittobyteacademy.com/blog/data-contracts-for-edge-ai-the-bug-class-that-doesnt-crash/)
introduces the problem,
[1.2](https://bittobyteacademy.com/blog/one-source-for-python-and-c-generating-shared-constants/)
covers Level 1,
[1.3](https://bittobyteacademy.com/blog/golden-vectors-testing-that-python-and-c-calculate-the-same-features/)
and [1.4](https://bittobyteacademy.com/blog/choosing-test-cases-and-proving-they-catch-real-bugs/)
cover Level 2, and
[1.5](https://bittobyteacademy.com/blog/refusing-to-build-on-guessed-values/)
covers Level 3.

## Level 1: constants

`data_contract.py` is the only file where the shared values are written by
hand. It checks that its own values are consistent, then generates three
files:

```
              data_contract.py
            (the only hand-written source)
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   contract.py  contract_data.h  contract_runtime.h
    (Python)         (C)              (C)
```

Every generated file records a hash of the contract it was made from.

```bash
python data_contract.py --check
```

This regenerates the three files in memory and compares them with the files
on disk. If a generated file was edited by hand, it reports `HAS DRIFTED`. If
the contract changed but nobody regenerated, it reports `OUT OF DATE`. Either
way it exits with status 1, which fails a CI job.

## Level 2: functions

A hash can confirm that `WINDOW_SAMPLES` is 100 on both sides. It cannot
confirm that two functions, written separately in Python and C, calculate
the same features.

**Golden vectors** solve this. `gen_golden.py` runs the Python reference,
`features.py`, on ten chosen input windows called scenes, and saves both the
inputs and the outputs. Both `features.py` and `features.c` must then
reproduce those outputs.

The vectors are written twice by the same generator: `golden_vectors.json`
for the Python test, because it is easy to read and review, and
`golden_vectors.h` for the C test, because a header needs no file system or
JSON parser and can run on a microcontroller.

### The ten scenes

Each scene is designed to catch a specific mistake.

| Scene | The mistake it is designed to catch |
|---|---|
| `all_zero` | Treating a reading of zero as missing |
| `channel_distinct` | Channels swapped, or results written to the wrong position |
| `scale_split` | Accelerometer and gyroscope conversions swapped |
| `alternating` | The wrong standard deviation formula (`ddof`) |
| `ramp` | Any two statistics confused with each other |
| `impulse` | Minimum and maximum mixed up, or a reading skipped |
| `negative` | Minimum and maximum started at zero |
| `some_missing` | Missing readings averaged in as real values |
| `all_missing` | A missing channel reported as zero |
| `realistic` | Anything the simpler scenes miss |

### Why the comparison has a tolerance

Python calculates in 64-bit floating point and the device in 32-bit, so the
two never agree to the last digit. A test that demands exact equality fails
all the time and soon gets switched off.

The contract sets `GOLDEN_REL_TOL = 1e-4` and `GOLDEN_ABS_TOL = 1e-3`, and a
value passes if it is within either. Measured on this code, the largest
float32 rounding difference is about `1.3e-7` relative, roughly 800 times
below the tolerance, and the smallest deliberate mistake (the standard
deviation formula) is about 0.5%, roughly 50 times above it.

The absolute tolerance is generous for small values: accelerometer features
in g are often below 1, so a 0.5% error in a small value can pass. In this
code every such mistake is still caught by the larger gyroscope values. Post
1.3 explains this in more detail.

## Proving the test works

A test that has never failed tells you nothing, because a useful golden test
and a useless one both report zero failures.

```bash
python sabotage.py           # run all ten
python sabotage.py --list    # describe each one
python sabotage.py ddof      # run just one
```

`sabotage.py` makes one deliberate change to `features.c`, rebuilds, checks
that the golden test now fails, and restores the original. Each change is a
mistake a capable engineer could realistically make.

| Sabotage | What it changes | Failures | Caught by |
|---|---|---|---|
| `ddof` | Uses the sample standard deviation formula | 22 | 5 scenes |
| `layout` | Writes results in the wrong order | 80 | all 10 scenes |
| `sentinel_zero` | Reports a missing channel as 0 | 4 | `all_missing` only |
| `sentinel_included` | Averages missing readings in as real values | 22 | `some_missing`, `all_missing` |
| `scale_swap` | Swaps the accelerometer and gyroscope conversions | 90 | 9 scenes |
| `scale_copy_paste` | Gives one gyroscope channel the accelerometer conversion | 30 | 9 scenes |
| `minmax_swap` | Swaps the minimum and maximum | 60 | 5 scenes |
| `minmax_init_zero` | Starts the minimum and maximum at 0 | 38 | 7 scenes |
| `off_by_one` | Skips the last reading of every window | 46 | 5 scenes |
| `zero_as_missing` | Treats a reading of 0 as missing | 33 | `all_zero`, `ramp` |

All ten are caught. The "caught by" column is the most useful part: it shows
which scenes do essential work. `sentinel_zero` is caught only by
`all_missing`, so removing that scene would let the mistake through.

If a sabotage is ever reported as `MISSED`, the golden vectors have a blind
spot, and the fix is to add a scene that targets it.

## Level 3: values nobody has decided yet

Some settings cannot be chosen until there is real data, so the contract
marks them as undecided:

```python
"DETECTION_THRESHOLD": None,      # <-- pick on a validation set
"ALERT_HOLD_S": None,             # <-- measure on the bench
```

For each value that is still `None`, the generator writes an `#error` into
`contract_runtime.h`, so any firmware that includes it will not compile:

```c
#error "DETECTION_THRESHOLD is unset in data_contract.py. Refusing to build on a guessed constant."
```

Once someone sets the value in the contract, the generator writes a normal
`#define` instead. A guessed value looks exactly like a decided one, so a
missing decision should stop the build rather than hide behind a default.

### Why there are two C headers

`contract_data.h` describes the shape of the data: channels, layout, units,
the missing-reading marker and the tolerances. It is fully decided and never
contains an `#error`. `contract_runtime.h` holds product settings that are
decided later.

`features.c` includes only `contract_data.h`, so the golden test builds and
runs from the first day, while the thresholds are still undecided. Only the
parts of the firmware that act on product settings should include
`contract_runtime.h`. This repository contains no device firmware, so nothing
here includes it yet.

## The tools must run in order

```bash
python data_contract.py       # 1. generate the contract files
python gen_golden.py          # 2. generate golden vectors from them
make run                      # 3. check the C code reproduces them
python test_contract.py       # 4. check the Python code does too
```

Step 2 imports the Python file that step 1 generates, and records the
contract's hash in both golden vector files. Whenever the contract changes,
run both steps again; `make generate` does exactly that.

If step 2 is forgotten, `python gen_golden.py --check` and the Python tests
report the mismatch, and the C test refuses to run, printing both hashes.
`make check` runs every check in the right order.

## Generated files

Five files are generated. **Never edit them by hand, but do commit them.**
The `--check` command compares the committed files with what the contract
produces now, so if they are not in the repository, there is nothing to
compare against. Committing them also means that a reviewer sees exactly
which values changed, in both languages, whenever the contract changes.

| Generated file | Generated by |
|---|---|
| `contract.py` | `data_contract.py` |
| `contract_data.h` | `data_contract.py` |
| `contract_runtime.h` | `data_contract.py` |
| `golden_vectors.json` | `gen_golden.py` |
| `golden_vectors.h` | `gen_golden.py` |

Everything else is written by hand: `data_contract.py`, `features.py`,
`features.c`, `features.h`, `gen_golden.py`, `test_golden.c`,
`test_contract.py`, `sabotage.py` and `Makefile`.

## What this does not do

- **It does not check that your data is good.** The checks make sure both
  sides process data the same way, and will carry a poor dataset through
  just as faithfully.
- **It does not check that your model is good.** There is no model here, on
  purpose.
- **It does not make the C code fast.** `features.c` is written to be easy
  to read. If you optimise it, the golden vectors tell you immediately if a
  result changed.

## Adapting it to your project

The example uses a six-channel motion sensor with four statistics per
channel, but nothing in the approach depends on that. Change `CHANNELS` and
`STATS` in the contract and the feature functions in `features.py` and
`features.c`, then write scenes that target the mistakes your own functions
could realistically contain.

Things worth checking in almost any project that trains in one language and
runs in another:

- the order of the sensor channels, especially when firmware fills a buffer
  in whatever order a loop reads the sensors
- units and scaling, from raw sensor readings to real units
- the marker for a missing reading, particularly when a cleared buffer
  supplies zeros
- quantisation scale and zero point, for models that run on 8-bit integers
- normalisation values calculated during training
- window length, step and overlap
- the standard deviation formula, where numpy and pandas use different
  defaults
- 64-bit numbers in training against 32-bit numbers on the device
- the order in which features are placed in the model's input
- how data structures are packed in memory and the byte order of data sent
  between devices

## Licence

Code in this folder is under the MIT licence. This README is under CC BY 4.0. See the
[top-level README](../README.md#licence) for details.
