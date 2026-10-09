"""
gen_golden.py -- freeze inputs and expected outputs, in both languages.

    python gen_golden.py           regenerate both golden files
    python gen_golden.py --check   verify neither was hand-edited (CI)

=============================================================================
WHAT A GOLDEN VECTOR IS
=============================================================================

    Freeze a set of specific inputs and the exact outputs the Python
    reference produces for them, commit those numbers, and make BOTH
    languages prove they reproduce them.

That is the whole idea. The C implementation is not "the same algorithm
written again and trusted". It is an implementation that has to produce these
specific numbers or the build fails.

An analogy. Translating a recipe, you can check the ingredient list matches --
that is the contract hash, level 1. But the METHOD, "fold gently until just
combined", can be mistranslated in ways that pass any word-by-word check, and
you find out when the cake is wrong. A golden vector is: bake it, weigh the
result, write down "500 g, 8 cm tall". Now the translation has to produce a
500 g, 8 cm cake. How the words came out does not matter.

THE OUTPUT IS THE SPECIFICATION.

=============================================================================
TWO FILES, ONE GENERATOR
=============================================================================

golden_vectors.json   for the Python tests -- readable, reviewable in a diff
golden_vectors.h      for the C test -- so the firmware test needs no JSON
                      parser, and can run on a target with no filesystem

Both come from this one generator, so they cannot disagree with each other.

=============================================================================
THE HASH STAMP
=============================================================================

Both files carry the contract hash they were generated against. If the
contract moves and these are not regenerated, the tests fail with the two
hashes side by side.

That check matters more than it looks: golden vectors certifying agreement on
a seam that has since moved are worse than no golden vectors at all, because
they report success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import (  # noqa: E402
    CONTRACT_HASH, CHANNELS, N_CHANNELS, STATS, N_STATS, N_FEATURES,
    WINDOW_SAMPLES, SENTINEL_MISSING, ACCEL_TICKS_PER_G,
)
from features import extract_features, feature_names  # noqa: E402

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "golden_vectors.json"
HEADER_PATH = HERE / "golden_vectors.h"

S = int(SENTINEL_MISSING)


# =========================================================================
# THE SCENES
# =========================================================================
# Each scene targets a specific way the seam can break. A scene that does not
# encode a failure mode is padding, and padding teaches people the test suite
# is decorative -- so every one below names the bug it catches.

def _blank(fill=0):
    return np.full((WINDOW_SAMPLES, N_CHANNELS), fill, dtype=np.int16)


def scene_all_zero():
    """Zero is a VALID reading, not absence. Must not be confused with the
    sentinel: every statistic here is 0.0, none is SENTINEL_MISSING."""
    return _blank(0)


def scene_channel_distinct():
    """Each channel a different constant. Catches channel reordering and any
    transposition of the feature layout: every channel's mean is unique, so
    a swap moves a recognisable value to the wrong slot."""
    w = _blank()
    for ch in range(N_CHANNELS):
        w[:, ch] = (ch + 1) * 1000
    return w


def scene_scale_split():
    """Identical ticks on every channel. Because accelerometer and gyroscope
    scale differently, correct output shows TWO distinct values. Swap the two
    factors and the numbers stay plausible but the groups exchange places."""
    return _blank(2048)


def scene_alternating():
    """+/-1000 ticks alternating.

    The ddof catcher. Population std (ddof=0) is exactly 1000 ticks; sample
    std (ddof=1) is 1000*sqrt(n/(n-1)), about 0.5% larger at n=100. Both are
    clean, plausible numbers -- which is exactly why this needs a frozen
    vector rather than an eyeball."""
    w = _blank()
    w[0::2, :] = 1000
    w[1::2, :] = -1000
    return w


def scene_ramp():
    """A different linear ramp per channel. Non-trivial mean, std, min and
    max simultaneously, all four distinct, so no statistic can be confused
    with another."""
    w = _blank()
    for ch in range(N_CHANNELS):
        w[:, ch] = np.arange(WINDOW_SAMPLES, dtype=np.int16) * (ch + 1)
    return w


def scene_impulse():
    """Flat, with one large spike. Catches min/max swaps, and any off-by-one
    that would clip the first or last sample out of the window."""
    w = _blank(100)
    w[WINDOW_SAMPLES // 2, :] = 20000
    return w


def scene_negative():
    """All values negative. Catches an accidental absolute value, and any
    min/max comparison initialised to 0 instead of the first valid sample."""
    w = _blank(-3000)
    w[0, :] = -9000
    return w


def scene_some_missing():
    """Sentinels mixed with real data in one window.

    The one that matters most. Excluded correctly, the mean is of the valid
    samples only. Included, -32768 is averaged into a real measurement and
    every statistic is wrecked -- silently, with a well-formed number."""
    w = _blank(500)
    w[10:20, :] = S
    w[70, :] = S
    return w


def scene_all_missing():
    """A completely absent channel. Every statistic must be SENTINEL_MISSING,
    not 0.0 -- zero would read as a valid measurement of a sensor at rest."""
    w = _blank(500)
    w[:, CHANNELS.index("gy")] = S
    return w


def scene_realistic():
    """Deterministic pseudo-random motion: nothing round, nothing symmetric,
    every feature a number that has to be computed rather than guessed. The
    scene that catches whatever the other eight did not."""
    rng = np.random.default_rng(20260831)
    w = np.zeros((WINDOW_SAMPLES, N_CHANNELS), dtype=np.int16)
    for ch in range(N_CHANNELS):
        base = ACCEL_TICKS_PER_G if ch < 3 else 0
        walk = np.cumsum(rng.normal(0, 200, WINDOW_SAMPLES))
        w[:, ch] = np.clip(base + walk, -32000, 32000).astype(np.int16)
    return w


SCENES = {
    "all_zero":         (scene_all_zero,
                         "zero is a valid reading, not the sentinel"),
    "channel_distinct": (scene_channel_distinct,
                         "channel order and feature layout"),
    "scale_split":      (scene_scale_split,
                         "accel vs gyro scale factors not swapped"),
    "alternating":      (scene_alternating,
                         "ddof: population vs sample std"),
    "ramp":             (scene_ramp,
                         "all four statistics distinct and non-trivial"),
    "impulse":          (scene_impulse,
                         "min/max, and no off-by-one at the window edges"),
    "negative":         (scene_negative,
                         "signed values; min/max not initialised to zero"),
    "some_missing":     (scene_some_missing,
                         "sentinels excluded, not averaged in"),
    "all_missing":      (scene_all_missing,
                         "a fully absent channel propagates the sentinel"),
    "realistic":        (scene_realistic,
                         "ordinary motion, nothing round or symmetric"),
}


def build():
    scenes = []
    for name, (fn, why) in SCENES.items():
        raw = fn()
        if raw.shape != (WINDOW_SAMPLES, N_CHANNELS):
            raise SystemExit(f"scene {name} has shape {raw.shape}, "
                             f"expected ({WINDOW_SAMPLES}, {N_CHANNELS})")
        scenes.append({
            "name": name,
            "catches": why,
            "input": raw.astype(int).tolist(),
            "expected": [float(x) for x in extract_features(raw)],
        })
    return {
        "contract_hash": CONTRACT_HASH,
        "window_samples": WINDOW_SAMPLES,
        "n_channels": N_CHANNELS,
        "n_features": N_FEATURES,
        "feature_names": feature_names(),
        "scenes": scenes,
    }


def render_json(g):
    return json.dumps(g, indent=1) + "\n"


def c_float(v):
    """
    A float as a valid C literal, round-tripping exactly.

    %.9g is enough digits to recover any float32 bit pattern, but it drops
    the decimal point on round numbers: 0.0 formats as "0", and "0f" is not
    a valid literal -- the f suffix needs a '.' or an exponent. Every value
    here would hit that, since half the golden scenes are deliberately made
    of round numbers.
    """
    if v != v or v in (float("inf"), float("-inf")):
        raise SystemExit(
            f"golden vector contains {v}, which has no portable C literal. "
            f"A NaN or infinity in an expected value means the Python "
            f"reference produced one -- fix that, do not encode it.")
    s = f"{v:.9g}"
    if "." not in s and "e" not in s and "E" not in s and "n" not in s:
        s += ".0"
    return s + "f"


def render_header(g):
    """
    C header form. Values are emitted with enough precision to round-trip a
    float exactly, so the C test compares against the same numbers the JSON
    holds rather than a rounded copy.
    """
    L = [
        "/*",
        " * GENERATED FILE -- DO NOT EDIT",
        " * Source: gen_golden.py",
        f" * Contract hash: {g['contract_hash']}",
        " *",
        " * Frozen inputs and the outputs the Python reference produces for",
        " * them. features.c must reproduce these within the contract's",
        " * tolerances or the test fails.",
        " *",
        " * Emitted as a header rather than read from JSON so the test needs",
        " * no parser and can run on a target with no filesystem.",
        " */",
        "",
        "#ifndef GOLDEN_VECTORS_H",
        "#define GOLDEN_VECTORS_H",
        "",
        "#include <stdint.h>",
        "",
        '#include "contract_data.h"',
        "",
        f'#define GOLDEN_CONTRACT_HASH "{g["contract_hash"]}"',
        f'#define N_GOLDEN_SCENES {len(g["scenes"])}',
        "",
        "typedef struct {",
        "    const char *name;",
        "    const char *catches;",
        "    const int16_t input[WINDOW_SAMPLES][N_CHANNELS];",
        "    const float expected[N_FEATURES];",
        "} golden_scene_t;",
        "",
        "static const char *const GOLDEN_FEATURE_NAMES[N_FEATURES] = {",
    ]
    L += [f'    "{n}",' for n in g["feature_names"]]
    L += ["};", "",
          "static const golden_scene_t GOLDEN_SCENES[N_GOLDEN_SCENES] = {"]

    for sc in g["scenes"]:
        L += ["    {",
              f'        .name = "{sc["name"]}",',
              f'        .catches = "{sc["catches"]}",',
              "        .input = {"]
        for row in sc["input"]:
            L.append("            {" + ",".join(str(v) for v in row) + "},")
        L += ["        },", "        .expected = {"]
        for i in range(0, len(sc["expected"]), 4):
            chunk = sc["expected"][i:i + 4]
            L.append("            " +
                     " ".join(c_float(v) + "," for v in chunk))
        L += ["        },", "    },"]

    L += ["};", "", "#endif /* GOLDEN_VECTORS_H */", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    g = build()
    files = {JSON_PATH: render_json(g), HEADER_PATH: render_header(g)}

    if args.check:
        bad = [p.name for p, want in files.items()
               if not p.exists() or p.read_text() != want]
        if bad:
            print(f"GOLDEN CHECK FAILED: {', '.join(bad)} drifted or missing",
                  file=sys.stderr)
            print("  Regenerate: python gen_golden.py", file=sys.stderr)
            raise SystemExit(1)
        print(f"  golden vectors in sync  ({len(g['scenes'])} scenes, "
              f"hash {g['contract_hash']})")
        return

    for p, text in files.items():
        p.write_text(text)
        print(f"  wrote {p.name}")
    print(f"\n  {len(g['scenes'])} scenes x {N_FEATURES} features = "
          f"{len(g['scenes']) * N_FEATURES} frozen comparisons")
    print(f"  contract hash {g['contract_hash']}")


if __name__ == "__main__":
    main()
