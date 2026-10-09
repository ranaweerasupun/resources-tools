"""
data_contract.py -- the single source of truth for the Python <-> C seam.

    python data_contract.py           regenerate every downstream file
    python data_contract.py --check   verify nothing has drifted (for CI)

=============================================================================
WHY THIS FILE EXISTS
=============================================================================

An edge AI system trains a model in one language and runs it in another.
Between them is a joint -- call it the SEAM. Both sides must agree exactly on
what a window of data is: how many samples, in what order, in what units,
with what meaning for "no reading".

Most bugs announce themselves. A crash, an assertion, a red test, a NaN. Seam
bugs do not. The model is fed something it was never trained on, and it
returns a confident, well-formed, completely wrong answer. Forever. Accuracy
drops a few points and you spend a fortnight blaming the model.

The naive fix is discipline: everyone remember to keep the two sides in sync.
That fails, always, because it relies on memory. The real fix has a shape:

    Do not REMEMBER to keep two things in sync.
    Make it IMPOSSIBLE for them to be out of sync without something failing
    loudly.

This applies at three levels, and each catches something the one below it
cannot:

    Level 1  a CONSTANT disagrees   -> one source, generated files, hash
    Level 2  a FUNCTION disagrees   -> golden vectors (see gen_golden.py)
    Level 3  a DECISION was skipped -> #error until someone decides

Level 1 is this file. Level 2 exists because:

    A constant that disagrees is caught by a hash.
    A function that disagrees is not.

You can hash WINDOW_SAMPLES = 100. You cannot hash "the routine that turns a
window into features". Two people write it two subtly different ways, both
files pass every check here, and the model is quietly fed the wrong thing.

=============================================================================
THE TWO HEADERS, AND WHY THEY ARE SEPARATE
=============================================================================

contract_data.h    the seam itself: channels, layout, units, sentinels.
                   Fully decided. Never emits #error.
contract_runtime.h product settings: thresholds, timeouts. Not yet decided.
                   Emits #error until someone measures them.

Fusing these means the seam cannot be tested until every product threshold is
frozen, which is backwards -- the seam is the thing you want pinned FIRST.
Keeping them apart is why the golden test in this directory compiles and runs
today, with DETECTION_THRESHOLD still unset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONTRACT_ID = "DC-001"
CONTRACT_REV = "r1.0"


# =========================================================================
# THE CONTRACT
# =========================================================================
# Everything the two languages must agree on. Nothing here is written down
# anywhere else; every other file is generated from this dict.

CONTRACT = {

    # --- sampling -------------------------------------------------------
    "SAMPLE_RATE_HZ": 50,
    "WINDOW_MS": 2000,

    # --- channels -------------------------------------------------------
    # ORDER IS LOAD-BEARING. The firmware fills its buffer in this order and
    # the model was trained on this order. Swap two entries and nothing
    # crashes; the model just reads pitch where it expects roll.
    "CHANNELS": ["ax", "ay", "az", "gx", "gy", "gz"],

    # --- feature layout -------------------------------------------------
    # Which statistics, and how the resulting array is laid out.
    #
    # "channel_major" means feature[ch * N_STATS + stat].
    # The alternative, feature[stat * N_CHANNELS + ch], holds exactly the
    # same values in a different order. That is the most invisible bug
    # available: every number is present and correct, just in the wrong slot.
    "STATS": ["mean", "std", "min", "max"],
    "FEATURE_LAYOUT": "channel_major",

    # ddof for every standard deviation on both sides of the seam.
    #
    # numpy.std defaults to ddof=0 (population).
    # pandas.Series.std defaults to ddof=1 (sample).
    # They differ by sqrt(n/(n-1)) -- about 0.5% at n=100.
    #
    # Nothing crashes. The features are quietly a fraction of a percent off,
    # in Python only, and the C side never agrees. This is a dialect
    # difference between two Python libraries, before C is even involved.
    # 0 is the form a naive C loop writes, so 0 is what we pin.
    "STD_DDOF": 0,

    # --- units ----------------------------------------------------------
    # The sensor reports signed 16-bit ADC ticks. These convert to physical
    # units. Both sides must scale identically or every threshold downstream
    # is meaningless.
    "ACCEL_TICKS_PER_G": 4096.0,
    "GYRO_TICKS_PER_DPS": 16.384,

    # --- the sentinel ---------------------------------------------------
    # What does a channel read when there is no valid measurement?
    #
    # 0.0 is what a zeroed struct gives you for free, and nobody types it
    # deliberately. It also means "a perfectly still sensor at rest", which
    # is a real, plausible reading. Using it for absence makes "no data"
    # indistinguishable from "a valid measurement of nothing happening".
    #
    # A sentinel must be a value the sensor CANNOT produce.
    "SENTINEL_MISSING": -32768.0,

    # --- golden vector tolerances ---------------------------------------
    # Python computes in float64; the device computes in float32. They will
    # never match to the last bit.
    #
    # A golden test demanding exact equality fails forever and gets switched
    # off within a week -- and a test that has been switched off is worse
    # than no test, because people believe it is running.
    #
    # Loose enough that float32 rounding passes. Tight enough that every
    # error that matters is caught: a wrong sentinel is off by thousands, a
    # transposed layout by orders of magnitude, a ddof mismatch by a
    # measurable fraction of a percent.
    "GOLDEN_REL_TOL": 1e-4,
    "GOLDEN_ABS_TOL": 1e-3,

    # --- DECISIONS NOT YET MADE (level 3) -------------------------------
    # None is deliberate, not an oversight. A missing decision must be a
    # LOUD failure, not a quiet default, because a guessed constant looks
    # exactly like a decided one six weeks later.
    #
    # These live in contract_runtime.h, which #errors until they are set.
    # contract_data.h -- the seam -- never does, which is why the golden
    # test builds and runs while these are still open.
    "DETECTION_THRESHOLD": None,      # <-- pick on a validation set
    "ALERT_HOLD_S": None,             # <-- measure on the bench
}


# --- derived values -------------------------------------------------------
# Computed, never hand-written, so they cannot contradict their inputs.

def derive(c):
    d = dict(c)
    d["N_CHANNELS"] = len(c["CHANNELS"])
    d["N_STATS"] = len(c["STATS"])
    d["WINDOW_SAMPLES"] = int(round(c["WINDOW_MS"] * c["SAMPLE_RATE_HZ"] / 1000))
    d["N_FEATURES"] = d["N_CHANNELS"] * d["N_STATS"]
    d["SAMPLE_INTERVAL_MS"] = 1000.0 / c["SAMPLE_RATE_HZ"]
    return d


def validate(d):
    """
    Internal consistency. A contract that disagrees with itself is worse
    than no contract, because both sides will faithfully implement the
    contradiction.
    """
    problems = []

    if d["WINDOW_SAMPLES"] < 2:
        problems.append(
            f"WINDOW_SAMPLES is {d['WINDOW_SAMPLES']}; a standard deviation "
            f"needs at least 2 samples.")

    if d["WINDOW_MS"] * d["SAMPLE_RATE_HZ"] % 1000 != 0:
        problems.append(
            f"WINDOW_MS={d['WINDOW_MS']} at {d['SAMPLE_RATE_HZ']} Hz is not a "
            f"whole number of samples. Rounding here means Python and C may "
            f"round differently.")

    if len(set(d["CHANNELS"])) != len(d["CHANNELS"]):
        problems.append("duplicate channel names")

    if d["STD_DDOF"] not in (0, 1):
        problems.append(f"STD_DDOF must be 0 or 1, got {d['STD_DDOF']}")

    if d["STD_DDOF"] >= d["WINDOW_SAMPLES"]:
        problems.append("STD_DDOF >= WINDOW_SAMPLES would divide by zero")

    if d["FEATURE_LAYOUT"] not in ("channel_major", "stat_major"):
        problems.append(f"unknown FEATURE_LAYOUT {d['FEATURE_LAYOUT']!r}")

    # The sentinel must be outside anything the sensor can report, or it is
    # ambiguous with real data.
    if -32768 < d["SENTINEL_MISSING"] < 32767:
        problems.append(
            f"SENTINEL_MISSING={d['SENTINEL_MISSING']} is inside the int16 "
            f"range the sensor can produce, so it cannot be distinguished "
            f"from a real reading.")

    if d["GOLDEN_REL_TOL"] <= 0 or d["GOLDEN_ABS_TOL"] <= 0:
        problems.append("golden tolerances must be positive")

    if problems:
        raise SystemExit("CONTRACT IS INTERNALLY INCONSISTENT:\n  - " +
                         "\n  - ".join(problems))
    return d


def contract_hash(d):
    """
    Stable fingerprint of the whole contract.

    sort_keys makes it independent of dict ordering, so reordering this file
    does not invent a spurious change. Every generated file carries this
    hash; if a data file and a code file disagree about it, they were built
    against different seams.
    """
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# =========================================================================
# generators
# =========================================================================

def _banner(comment, hash_):
    c = comment
    return (f"{c} GENERATED FILE -- DO NOT EDIT\n"
            f"{c} Source: data_contract.py   "
            f"Contract: {CONTRACT_ID} {CONTRACT_REV}   Hash: {hash_}\n"
            f"{c}\n"
            f"{c} Editing this by hand puts it out of sync with its\n"
            f"{c} counterpart in the other language, which is exactly the\n"
            f"{c} failure this file exists to prevent. Edit data_contract.py\n"
            f"{c} and regenerate ALL of them.\n")


def gen_python(d, h):
    L = ['"""', _banner("", h).rstrip(), '"""', "",
         f'CONTRACT_HASH = "{h}"', ""]

    L += ["# --- sampling ---"]
    for k in ("SAMPLE_RATE_HZ", "WINDOW_MS", "WINDOW_SAMPLES"):
        L.append(f"{k:<22}= {d[k]}")
    L.append(f"{'SAMPLE_INTERVAL_MS':<22}= {d['SAMPLE_INTERVAL_MS']!r}")

    L += ["", "# --- channels: ORDER IS LOAD-BEARING ---",
          f"{'CHANNELS':<22}= {d['CHANNELS']!r}",
          f"{'N_CHANNELS':<22}= {d['N_CHANNELS']}"]

    L += ["", "# --- feature layout ---",
          f"{'STATS':<22}= {d['STATS']!r}",
          f"{'N_STATS':<22}= {d['N_STATS']}",
          f"{'N_FEATURES':<22}= {d['N_FEATURES']}",
          f"{'FEATURE_LAYOUT':<22}= {d['FEATURE_LAYOUT']!r}",
          f"{'STD_DDOF':<22}= {d['STD_DDOF']}"]

    L += ["", "# --- units ---"]
    for k in ("ACCEL_TICKS_PER_G", "GYRO_TICKS_PER_DPS"):
        L.append(f"{k:<22}= {d[k]!r}")

    L += ["", "# --- sentinel: a value the sensor cannot produce ---",
          f"{'SENTINEL_MISSING':<22}= {d['SENTINEL_MISSING']!r}"]

    L += ["", "# --- golden vector tolerances ---"]
    for k in ("GOLDEN_REL_TOL", "GOLDEN_ABS_TOL"):
        L.append(f"{k:<22}= {d[k]!r}")

    L += ["", "# --- NOT YET DECIDED. None on purpose; see data_contract.py ---"]
    for k in ("DETECTION_THRESHOLD", "ALERT_HOLD_S"):
        L.append(f"{k:<22}= {d[k]!r}")

    L += ["", "",
          "def feature_index(channel, stat):",
          '    """',
          "    Index of one (channel, stat) pair in the feature vector.",
          "",
          "    Both languages compute this the same way from the same",
          "    layout constant, so neither can hand-roll it differently.",
          '    """',
          "    ch = CHANNELS.index(channel) if isinstance(channel, str) else channel",
          "    st = STATS.index(stat) if isinstance(stat, str) else stat",
          '    if FEATURE_LAYOUT == "channel_major":',
          "        return ch * N_STATS + st",
          "    return st * N_CHANNELS + ch",
          ""]
    return "\n".join(L)


def gen_contract_data_h(d, h):
    """The seam. Fully decided, so this NEVER emits #error."""
    L = ["/*", _banner(" *", h).rstrip(), " */", "",
         "#ifndef CONTRACT_DATA_H", "#define CONTRACT_DATA_H", "",
         f'#define CONTRACT_HASH "{h}"', ""]

    L += ["/* --- sampling --- */",
          f"#define SAMPLE_RATE_HZ      {d['SAMPLE_RATE_HZ']}",
          f"#define WINDOW_MS           {d['WINDOW_MS']}",
          f"#define WINDOW_SAMPLES      {d['WINDOW_SAMPLES']}", ""]

    L += ["/* --- channels: ORDER IS LOAD-BEARING. The buffer is filled",
          "   in this order and the model was trained on this order. --- */",
          f"#define N_CHANNELS          {d['N_CHANNELS']}"]
    for i, ch in enumerate(d["CHANNELS"]):
        L.append(f"#define CH_{ch.upper():<16} {i}")
    L.append("")
    names = ", ".join(f'"{c}"' for c in d["CHANNELS"])
    L += [f"static const char *const CHANNEL_NAMES[N_CHANNELS] = {{ {names} }};",
          ""]

    L += ["/* --- feature layout --- */",
          f"#define N_STATS             {d['N_STATS']}",
          f"#define N_FEATURES          {d['N_FEATURES']}"]
    for i, st in enumerate(d["STATS"]):
        L.append(f"#define STAT_{st.upper():<14} {i}")
    L.append(f"#define STD_DDOF            {d['STD_DDOF']}")
    L.append("")
    if d["FEATURE_LAYOUT"] == "channel_major":
        idx = "((ch) * N_STATS + (stat))"
    else:
        idx = "((stat) * N_CHANNELS + (ch))"
    L += [f"/* layout: {d['FEATURE_LAYOUT']} */",
          f"#define FEATURE_INDEX(ch, stat) {idx}", ""]

    L += ["/* --- units --- */",
          f"#define ACCEL_TICKS_PER_G   {d['ACCEL_TICKS_PER_G']!r}f",
          f"#define GYRO_TICKS_PER_DPS  {d['GYRO_TICKS_PER_DPS']!r}f", ""]

    L += ["/* --- sentinel: a value the sensor cannot produce.",
          "   NOT 0.0 -- zero is what a zeroed struct gives you for free,",
          "   and it is also a valid reading of a sensor at rest. --- */",
          f"#define SENTINEL_MISSING    {d['SENTINEL_MISSING']!r}f", ""]

    L += ["/* --- golden vector tolerances --- */",
          f"#define GOLDEN_REL_TOL      {d['GOLDEN_REL_TOL']!r}f",
          f"#define GOLDEN_ABS_TOL      {d['GOLDEN_ABS_TOL']!r}f", ""]

    L += ["#endif /* CONTRACT_DATA_H */", ""]
    return "\n".join(L)


def gen_contract_runtime_h(d, h):
    """
    Product settings. Emits #error for anything still undecided, so the
    firmware refuses to build on a guess.
    """
    L = ["/*", _banner(" *", h).rstrip(), " */", "",
         "#ifndef CONTRACT_RUNTIME_H", "#define CONTRACT_RUNTIME_H", "",
         "/* Values still open in the contract appear below as #error.",
         "   That is deliberate. A guessed constant is worse than no",
         "   constant, because a guess looks exactly like a decision. */", ""]

    for k in ("DETECTION_THRESHOLD", "ALERT_HOLD_S"):
        v = d[k]
        if v is None:
            L += [f'#error "{k} is unset in data_contract.py. Refusing to '
                  f'build on a guessed constant."', ""]
        else:
            L += [f"#define {k:<20}{v!r}f", ""]

    L += ["#endif /* CONTRACT_RUNTIME_H */", ""]
    return "\n".join(L)


TARGETS = {
    "contract.py": gen_python,
    "contract_data.h": gen_contract_data_h,
    "contract_runtime.h": gen_contract_runtime_h,
}


def build():
    d = validate(derive(CONTRACT))
    h = contract_hash(d)
    return d, h, {name: fn(d, h) for name, fn in TARGETS.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify files on disk match; exit 1 if not (CI)")
    args = ap.parse_args()

    d, h, files = build()

    if args.check:
        drifted = []
        for name, want in files.items():
            p = HERE / name
            if not p.exists():
                drifted.append(f"{name} is MISSING")
            elif p.read_text() != want:
                drifted.append(f"{name} HAS DRIFTED -- someone hand-edited a "
                               f"generated file")
        if drifted:
            print(f"CONTRACT CHECK FAILED (hash {h})", file=sys.stderr)
            for x in drifted:
                print(f"  x {x}", file=sys.stderr)
            print("\n  Regenerate BOTH sides; do not patch one:",
                  file=sys.stderr)
            print("      python data_contract.py", file=sys.stderr)
            raise SystemExit(1)
        print(f"  contract {CONTRACT_ID} {CONTRACT_REV}  hash {h}  "
              f"-- {len(files)} generated files in sync")
        return

    for name, text in files.items():
        (HERE / name).write_text(text)
        print(f"  wrote {name}")
    print(f"\n  contract {CONTRACT_ID} {CONTRACT_REV}  hash {h}")
    print(f"  {d['N_CHANNELS']} channels x {d['N_STATS']} stats "
          f"= {d['N_FEATURES']} features, "
          f"{d['WINDOW_SAMPLES']}-sample windows")

    open_ = [k for k in ("DETECTION_THRESHOLD", "ALERT_HOLD_S")
             if d[k] is None]
    if open_:
        print(f"\n  {len(open_)} decision(s) still open: {', '.join(open_)}")
        print(f"  contract_runtime.h will #error until they are set. "
              f"That is intended.")
    print(f"\n  Golden vectors are stamped with the contract hash, so run")
    print(f"  gen_golden.py next or they will certify a seam that has moved.")


if __name__ == "__main__":
    main()
