"""
sabotage.py -- break the C implementation on purpose, ten ways.

    python sabotage.py            run them all
    python sabotage.py --list     show what each one does
    python sabotage.py ddof       run one

=============================================================================
WHY THIS EXISTS
=============================================================================

A test suite that has never failed is decoration.

You cannot tell a working golden test from a vacuous one by reading it. Both
print the same thing. The only way to know the check bites is to break the
implementation deliberately and watch it catch you.

Every sabotage below is a plausible edit. Not one is a typo or a deliberate
absurdity -- each is a line a competent engineer might write on a Tuesday,
and each produces confident, well-formed, wrong numbers. That is the bug
class this whole directory exists to defend against.

The script patches features.c in memory, rebuilds, runs the golden test,
confirms it FAILS, and restores the original. If any sabotage passes, that is
reported as a hole in the scene coverage -- a bug the suite cannot see.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "features.c"


# (name, description, why_plausible, old, new)
SABOTAGES = [
    (
        "ddof",
        "population -> sample standard deviation",
        "numpy.std is ddof=0, pandas.Series.std is ddof=1. Write the Python "
        "reference with pandas by accident and every std is 0.5% off, in one "
        "language only.",
        "const int denom = n - STD_DDOF;",
        "const int denom = n - 1;",
    ),
    (
        "layout",
        "transpose the feature vector",
        "Every value present, every value correct, every value in the wrong "
        "slot. The most invisible bug available -- no statistic is wrong, so "
        "nothing looks odd in isolation.",
        "out[FEATURE_INDEX(ch, STAT_MEAN)] = (float)mean;",
        "out[STAT_MEAN * N_CHANNELS + ch] = (float)mean;",
    ),
    (
        "sentinel_zero",
        "return 0.0 for a fully-missing channel",
        "Zero is what a zeroed struct gives you for free; nobody types it "
        "deliberately. It also reads as a valid measurement of a sensor "
        "sitting perfectly still.",
        "out[FEATURE_INDEX(ch, st)] = SENTINEL_MISSING;",
        "out[FEATURE_INDEX(ch, st)] = 0.0f;",
    ),
    (
        "sentinel_included",
        "average sentinels in as if they were readings",
        "Forgetting the skip is a one-line omission. -32768 then gets scaled "
        "to -8 g and averaged into real data, which is plausible enough that "
        "nothing downstream complains.",
        "            if ((float)raw[i][ch] == SENTINEL_MISSING) {\n"
        "                continue;\n"
        "            }\n"
        "            const float v = (float)raw[i][ch] * scale;",
        "            const float v = (float)raw[i][ch] * scale;",
    ),
    (
        "scale_swap",
        "swap the accelerometer and gyroscope scale factors",
        "Two constants of similar shape, one letter apart in the name. Every "
        "value stays in a believable range.",
        "    case CH_AX:\n"
        "    case CH_AY:\n"
        "    case CH_AZ:\n"
        "        return 1.0f / ACCEL_TICKS_PER_G;",
        "    case CH_AX:\n"
        "    case CH_AY:\n"
        "    case CH_AZ:\n"
        "        return 1.0f / GYRO_TICKS_PER_DPS;",
    ),
    (
        "scale_copy_paste",
        "give one gyroscope channel the accelerometer scale",
        "A copy-paste slip in a switch statement. One axis is converted with "
        "the wrong factor, and its values still look like real readings.",
        "    case CH_GX:\n"
        "    case CH_GY:\n"
        "    case CH_GZ:\n"
        "        return 1.0f / GYRO_TICKS_PER_DPS;",
        "    case CH_GY:\n"
        "    case CH_GZ:\n"
        "        return 1.0f / GYRO_TICKS_PER_DPS;\n"
        "    case CH_GX:\n"
        "        return 1.0f / ACCEL_TICKS_PER_G;",
    ),
    (
        "minmax_swap",
        "swap min and max",
        "Two adjacent lines differing by one character.",
        "        out[FEATURE_INDEX(ch, STAT_MIN)]  = vmin;\n"
        "        out[FEATURE_INDEX(ch, STAT_MAX)]  = vmax;",
        "        out[FEATURE_INDEX(ch, STAT_MIN)]  = vmax;\n"
        "        out[FEATURE_INDEX(ch, STAT_MAX)]  = vmin;",
    ),
    (
        "minmax_init_zero",
        "initialise min/max to 0 instead of the first valid sample",
        "The textbook way to write a min/max loop, and correct for data that "
        "straddles zero. On all-negative data the max is silently 0.",
        "            if (n == 0 || v < vmin) { vmin = v; }\n"
        "            if (n == 0 || v > vmax) { vmax = v; }",
        "            if (v < vmin) { vmin = v; }\n"
        "            if (v > vmax) { vmax = v; }",
    ),
    (
        "off_by_one",
        "drop the last sample of every window",
        "A loop bound that stops one reading early. The classic off-by-one "
        "mistake, and every statistic it touches stays plausible.",
        "        for (int i = 0; i < WINDOW_SAMPLES; i++) {\n"
        "            if ((float)raw[i][ch] == SENTINEL_MISSING) {\n"
        "                continue;\n"
        "            }\n"
        "            const float v = (float)raw[i][ch] * scale;",
        "        for (int i = 0; i < WINDOW_SAMPLES - 1; i++) {\n"
        "            if ((float)raw[i][ch] == SENTINEL_MISSING) {\n"
        "                continue;\n"
        "            }\n"
        "            const float v = (float)raw[i][ch] * scale;",
    ),
    (
        "zero_as_missing",
        "treat a reading of zero as missing",
        "Code that started with a zeroed buffer often treats 0 as 'no data'. "
        "But 0 is a real reading: an axis at rest, or a gyroscope that is "
        "not rotating. Those readings are then silently thrown away.",
        "            if ((float)raw[i][ch] == SENTINEL_MISSING) {\n"
        "                continue;\n"
        "            }\n"
        "            const float v = (float)raw[i][ch] * scale;",
        "            if (raw[i][ch] == 0 ||\n"
        "                (float)raw[i][ch] == SENTINEL_MISSING) {\n"
        "                continue;\n"
        "            }\n"
        "            const float v = (float)raw[i][ch] * scale;",
    ),
]


def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=HERE, capture_output=True,
                          text=True, **kw)


def apply_one(name, desc, why, old, new, verbose):
    src = TARGET.read_text()

    if src.count(old) != 1:
        return ("SKIP", f"anchor text appears {src.count(old)} times "
                        f"(expected exactly 1) -- features.c has changed and "
                        f"this sabotage needs updating")

    backup = Path(tempfile.mkdtemp(prefix="sabotage-")) / "features.c.orig"
    backup.write_text(src)

    try:
        TARGET.write_text(src.replace(old, new))

        build = run("make -B test_golden")
        if build.returncode != 0:
            # A sabotage that will not compile has proved nothing about the
            # golden vectors -- the compiler caught it, which is a different
            # (and better) mechanism.
            return ("COMPILE", "did not compile; the compiler caught this "
                               "one before the test could")

        result = run("./test_golden")
        if result.returncode == 0:
            return ("MISSED", "golden test PASSED on sabotaged code -- "
                              "no scene covers this failure mode")

        lines = [l for l in result.stdout.splitlines() if l.startswith("FAIL")]
        scenes = [l.split()[1] for l in lines]
        tail = [l for l in result.stdout.splitlines() if "comparisons" in l]
        if verbose:
            print(result.stdout)
        return ("CAUGHT", f"{tail[0].strip() if tail else ''} "
                          f"| scenes: {', '.join(scenes) or 'n/a'}")
    finally:
        # Restore unconditionally, including on Ctrl-C. Leaving a sabotaged
        # features.c behind would be a genuinely nasty thing for this script
        # to do.
        TARGET.write_text(backup.read_text())
        shutil.rmtree(backup.parent, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*", help="run only these sabotages")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show the full failure output for each")
    args = ap.parse_args()

    if args.list:
        for name, desc, why, _, _ in SABOTAGES:
            print(f"\n  {name}\n    {desc}\n    {why}")
        return

    chosen = [s for s in SABOTAGES if not args.only or s[0] in args.only]
    if not chosen:
        raise SystemExit(f"no sabotage matched {args.only}")

    # Establish that the suite passes BEFORE breaking anything. Otherwise a
    # "caught" result might just be a pre-existing failure.
    if run("make -B test_golden").returncode != 0:
        raise SystemExit("baseline build failed -- fix that first")
    if run("./test_golden").returncode != 0:
        raise SystemExit("baseline golden test already fails -- fix that "
                         "before sabotaging anything")
    print(f"  baseline: golden test passes\n")

    print(f"  {'sabotage':<20}{'result':<10}detail")
    print("  " + "-" * 76)

    verdicts = []
    for name, desc, why, old, new in chosen:
        status, detail = apply_one(name, desc, why, old, new, args.verbose)
        verdicts.append((name, status))
        print(f"  {name:<20}{status:<10}{detail}")

    missed = [n for n, s in verdicts if s == "MISSED"]
    skipped = [n for n, s in verdicts if s == "SKIP"]

    print()
    caught = sum(1 for _, s in verdicts if s in ("CAUGHT", "COMPILE"))
    print(f"  {caught}/{len(verdicts)} sabotages detected")

    if missed:
        print(f"\n  NOT DETECTED: {', '.join(missed)}")
        print("  Each of these is a real bug the golden vectors cannot see.")
        print("  Add a scene that distinguishes it, then re-run.")
    if skipped:
        print(f"\n  SKIPPED (anchor text not found): {', '.join(skipped)}")

    # Confirm we left the tree exactly as we found it.
    if run("make -B test_golden").returncode == 0 and \
            run("./test_golden").returncode == 0:
        print("\n  features.c restored; golden test passes again")
    else:
        print("\n  !! features.c may not have been restored cleanly. "
              "Check `git diff`.")
        raise SystemExit(1)

    raise SystemExit(1 if missed else 0)


if __name__ == "__main__":
    main()
