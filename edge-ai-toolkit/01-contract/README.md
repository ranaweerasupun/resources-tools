# 01 — Data contracts for edge AI

Companion code for **post 1** of the Edge AI series.

Everything here runs on a laptop. No board, no sensor, no dataset, no SDK —
just Python and a C compiler.

```bash
make generate     # regenerate every generated file from the contract
make run          # build and run the C golden test
make check        # the full three-level check, as CI would run it
python sabotage.py   # break the C code ten ways, prove each is caught
```

---

## The problem

An edge AI system trains a model in one language and runs it in another.
Between them is a joint — the **seam**. Both sides must agree exactly on what
a window of data is: how many samples, in what order, in what units, with
what meaning for "no reading".

Most bugs announce themselves. A crash, an assertion, a red test, a `NaN`.
Seam bugs do not. The model is fed something it was never trained on and
returns a confident, well-formed, completely wrong answer. Forever.

The naive fix is discipline — everyone remember to keep the two sides in
sync. That fails, always, because it relies on memory. The real fix has a
shape:

> Do not **remember** to keep two things in sync.
> Make it **impossible** for them to be out of sync without something failing
> loudly.

That applies at three levels, and each catches what the one below it cannot.

| Level | What disagrees | Mechanism | Files |
|---|---|---|---|
| 1 | a **constant** | one source, generated files, hash + `--check` | `data_contract.py` |
| 2 | a **function** | golden vectors | `gen_golden.py`, `test_golden.c` |
| 3 | a **decision nobody made** | `#error` until it is decided | `contract_runtime.h` |

---

## Level 1 — constants

`data_contract.py` is the only place the numbers exist. It generates the
files both languages use, and every generated file carries a hash of the
whole contract.

```
              data_contract.py          the only hand-written source
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  contract.py  contract_data.h  contract_runtime.h
   (Python)         (C)              (C)
```

```bash
python data_contract.py --check
```

Regenerates everything in memory and compares against disk. Hand-edit a
generated file and it says so and exits non-zero.

---

## Level 2 — functions

This is the part most people skip, and it is the reason the directory exists.

> **A constant that disagrees is caught by a hash.**
> **A function that disagrees is not.**

You can hash `WINDOW_SAMPLES = 100`. You cannot hash "the routine that turns
a window into features". Two people write it two subtly different ways, both
files pass every level-1 check, and the model is quietly fed the wrong thing.

A **golden vector** is: freeze specific inputs and the exact outputs the
Python reference produces for them, commit those numbers, and make *both*
languages prove they reproduce them.

Think of translating a recipe. You can check the ingredient list matches —
that is level 1. But the *method*, "fold gently until just combined", can be
mistranslated in ways that pass any word-by-word check, and you find out when
the cake is wrong. A golden vector is: bake it, weigh it, write down "500 g,
8 cm tall". The translation now has to produce a 500 g, 8 cm cake.

**The output is the specification.**

Ten scenes are frozen in `golden_vectors.json`. Each targets a specific way
the seam can break — a scene that does not encode a failure mode is padding,
and padding teaches people the suite is decorative.

| Scene | The bug it catches |
|---|---|
| `all_zero` | zero is a valid reading, not the sentinel |
| `channel_distinct` | channel order and feature layout |
| `scale_split` | accelerometer vs gyroscope factors swapped |
| `alternating` | `ddof` — population vs sample standard deviation |
| `ramp` | all four statistics distinct and non-trivial |
| `impulse` | min/max, and no off-by-one at the window edges |
| `negative` | signed values; min/max not initialised to zero |
| `some_missing` | sentinels excluded, not averaged in |
| `all_missing` | a fully absent channel propagates the sentinel |
| `realistic` | ordinary motion, nothing round or symmetric |

### Two files, one generator

`gen_golden.py` writes the same vectors twice — `golden_vectors.json` for the
Python tests (readable, reviewable in a diff) and `golden_vectors.h` for the
C test (no JSON parser, runs on a target with no filesystem). One generator,
so they cannot disagree with each other.

### Why not exact equality

Python computes in float64, the device in float32. They will never match to
the last bit. A golden test demanding exact equality fails forever and gets
switched off within a week — and a switched-off test is worse than no test,
because people believe it is running.

So the contract carries `GOLDEN_REL_TOL = 1e-4` and `GOLDEN_ABS_TOL = 1e-3`.
Loose enough for float32 rounding. Tight enough that a wrong sentinel (off by
thousands), a transposed layout (off by orders of magnitude) and a `ddof`
mismatch (off by ~0.5%) all fall well outside.

---

## Level 3 — decisions nobody made

Some values are `None` on purpose:

```python
"DETECTION_THRESHOLD": None,      # <-- pick on a validation set
"ALERT_HOLD_S": None,             # <-- measure on the bench
```

These are not oversights. A missing decision must be a **loud** failure, not
a quiet default, so the generated header emits:

```c
#error "DETECTION_THRESHOLD is unset in data_contract.py. Refusing to build on a guessed constant."
```

The firmware will not compile until someone decides. **A guessed constant is
worse than no constant, because a guess looks exactly like a decision.**

### Why there are two headers

`contract_data.h` is the seam — channels, layout, units, sentinels. Fully
decided, never `#error`s. `contract_runtime.h` is product settings —
thresholds, hold times. Not decided, so it does.

Fusing them means the seam cannot be tested until every product threshold is
frozen, which is backwards: the seam is what you want pinned **first**.
Keeping them apart is why the golden test runs today with
`DETECTION_THRESHOLD` still open. `features.c` includes only
`contract_data.h`.

---

## Proof it works

A test suite that has never failed is decoration. You cannot tell a working
golden test from a vacuous one by reading it — both print the same thing.

```bash
python sabotage.py
```

Breaks `features.c` ten ways, rebuilds, confirms the golden test **fails**
each time, and restores the original. Every sabotage is a line a competent
engineer might write on a Tuesday:

```
  ddof                CAUGHT    22 failures | alternating, ramp, impulse, negative, realistic
  layout              CAUGHT    72 failures | channel_distinct, scale_split, ...
  sentinel_zero       CAUGHT     4 failures | all_missing
  sentinel_included   CAUGHT    22 failures | some_missing, all_missing
  scale_swap          CAUGHT    90 failures | channel_distinct, scale_split, ...
  channel_swap        CAUGHT    30 failures | channel_distinct, scale_split, ...
  minmax_swap         CAUGHT    60 failures | alternating, ramp, impulse, ...
  minmax_init_zero    CAUGHT    38 failures | channel_distinct, scale_split, ...
  off_by_one          CAUGHT    46 failures | alternating, ramp, impulse, ...
  scale_after_stats   CAUGHT    11 failures | channel_distinct, scale_split, ...

  10/10 sabotages detected
```

The scene attribution is the useful part. `sentinel_zero` is caught by
exactly one scene — so `all_missing` is load-bearing, not decorative. Delete
it and that bug becomes invisible.

If a sabotage is ever reported as `MISSED`, that is a real bug the vectors
cannot see, and the fix is a new scene.

---

## Run order is a dependency chain

```bash
python data_contract.py       # 1. must be first
python gen_golden.py          # 2. imports what step 1 generated
make run                      # 3. the C side proves it agrees
python test_contract.py       # 4. the Python side proves it too
```

Step 2 **imports** step 1's output and stamps the contract hash into both
golden files. Change `data_contract.py` and you must re-run both. Forget, and
`test_golden_was_generated_against_this_contract` fails with the two hashes
side by side — because **golden vectors certifying agreement on a seam that
has since moved are worse than none.**

---

## Generated vs hand-written

**Five files are generated. Never edit them.** Commit them anyway — do not
gitignore them. They are committed precisely so drift shows up as a diff in a
pull request. That is the entire mechanism.

| Generated | By |
|---|---|
| `contract.py` | `data_contract.py` |
| `contract_data.h` | `data_contract.py` |
| `contract_runtime.h` | `data_contract.py` |
| `golden_vectors.json` | `gen_golden.py` |
| `golden_vectors.h` | `gen_golden.py` |

Everything else is hand-written: `data_contract.py`, `features.py`,
`features.c`, `features.h`, `gen_golden.py`, `test_golden.c`,
`test_contract.py`, `sabotage.py`, `Makefile`.

---

## What this does not do

- **It does not check your data is good.** A perfectly pinned seam happily
  carries a terrible dataset.
- **It does not check your model is good.** No model here, deliberately.
- **It does not make the C fast.** `features.c` is written for clarity.

---

## Adapting it

The example is a six-channel IMU windowed into four statistics per channel,
but nothing about the machinery is IMU-specific. Change `CHANNELS`, `STATS`
and the body of `extract_features` / `features_extract`, then write scenes
that break *your* function in the ways it can actually break.

The generic catalogue of things that silently disagree across a seam:

- channel order, when firmware fills a buffer in loop order
- units and scaling — raw ticks against physical units
- sentinels for absent data, where a zeroed struct hands you `0`
- quantization scale and zero-point for int8 models
- normalization statistics baked in at training time
- window length, stride, overlap
- `ddof` — numpy defaults to 0, pandas to 1
- float64 in training, float32 on device
- feature layout, `[channel][stat]` against `[stat][channel]`
- struct packing and endianness on the wire
