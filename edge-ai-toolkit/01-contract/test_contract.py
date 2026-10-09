"""
test_contract.py -- the Python half of the check.

    python test_contract.py        run directly, no pytest needed
    pytest test_contract.py        or under pytest, if you have it

Covers what the C test cannot: that the Python reference itself still
reproduces the frozen vectors, that the generated files have not drifted,
and that the contract is internally consistent.

The C test (test_golden.c) covers the other direction. Both are needed --
golden vectors only mean something if BOTH implementations are held to them.
Holding just one side to them proves nothing about agreement.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import contract                                    # noqa: E402
import data_contract                               # noqa: E402
from features import extract_features, feature_names, scale_window  # noqa: E402


def test_contract_files_have_not_drifted():
    """Level 1. Someone hand-edited a generated file, or forgot to
    regenerate after changing data_contract.py."""
    r = subprocess.run([sys.executable, "data_contract.py", "--check"],
                       cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or r.stdout


def test_contract_is_internally_consistent():
    """The validator is not just decoration: feed it a contradiction and it
    must refuse. A contract that disagrees with itself is worse than none,
    because both sides faithfully implement the contradiction."""
    bad = dict(data_contract.CONTRACT)
    bad["WINDOW_MS"] = 33          # 33 ms at 50 Hz is 1.65 samples
    try:
        data_contract.validate(data_contract.derive(bad))
    except SystemExit:
        pass
    else:
        raise AssertionError(
            "validate() accepted a window length that is not a whole number "
            "of samples -- Python and C could round it differently")


def test_golden_vectors_have_not_drifted():
    """Level 2, part one."""
    r = subprocess.run([sys.executable, "gen_golden.py", "--check"],
                       cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or r.stdout


def test_golden_was_generated_against_this_contract():
    """
    The stamp check.

    Golden vectors certifying agreement on a seam that has since moved are
    worse than no vectors, because they report success. gen_golden.py imports
    what data_contract.py generates, so changing the contract means running
    BOTH -- and forgetting shows up here with the two hashes side by side.
    """
    g = json.loads((HERE / "golden_vectors.json").read_text())
    assert g["contract_hash"] == contract.CONTRACT_HASH, (
        f"golden_vectors.json was frozen against {g['contract_hash']}, "
        f"but the contract is now {contract.CONTRACT_HASH}. "
        f"Run: python data_contract.py && python gen_golden.py")


def test_python_reproduces_the_golden_vectors():
    """
    Level 2, part two: the reference must still agree with itself.

    This is not circular. The vectors are committed; this catches an edit to
    features.py that changes behaviour without anyone regenerating them --
    which is the Python-side equivalent of the C drift the golden test
    catches.
    """
    g = json.loads((HERE / "golden_vectors.json").read_text())
    rel, abs_ = contract.GOLDEN_REL_TOL, contract.GOLDEN_ABS_TOL
    names = feature_names()

    for scene in g["scenes"]:
        got = extract_features(np.array(scene["input"], dtype=np.int16))
        want = np.array(scene["expected"], dtype=np.float64)
        bad = ~np.isclose(got, want, rtol=rel, atol=abs_)
        assert not bad.any(), (
            f"scene {scene['name']}: "
            + ", ".join(f"{names[i]} got {got[i]:.6g} want {want[i]:.6g}"
                        for i in np.where(bad)[0][:5]))


def test_feature_layout_is_a_bijection():
    """
    Every (channel, stat) pair maps to a distinct slot, and together they
    fill the vector exactly.

    A layout that collides silently overwrites one feature with another, and
    a layout that leaves gaps feeds uninitialised memory to the model. Both
    produce a well-formed vector.
    """
    seen = [contract.feature_index(ch, st)
            for ch in range(contract.N_CHANNELS)
            for st in range(contract.N_STATS)]
    assert sorted(seen) == list(range(contract.N_FEATURES)), (
        "feature_index is not a bijection onto [0, N_FEATURES)")


def test_sentinel_is_not_scaled():
    """
    A sentinel is a marker, not a measurement.

    Scaling -32768 by the accelerometer factor gives -8.0 g, which is a
    perfectly plausible reading -- and would then be treated as one.
    """
    raw = np.full((contract.WINDOW_SAMPLES, contract.N_CHANNELS), 500,
                  dtype=np.int16)
    raw[5, 0] = int(contract.SENTINEL_MISSING)
    out = scale_window(raw)
    assert out[5, 0] == contract.SENTINEL_MISSING, (
        f"the sentinel was scaled to {out[5, 0]}, which is inside the range "
        f"of a real measurement")


def test_missing_channel_does_not_become_zero():
    """
    Zero is a valid reading of a sensor at rest. Using it for absence makes
    'no data' indistinguishable from 'nothing happening'.
    """
    raw = np.full((contract.WINDOW_SAMPLES, contract.N_CHANNELS), 500,
                  dtype=np.int16)
    raw[:, 2] = int(contract.SENTINEL_MISSING)
    feats = extract_features(raw)
    for st in range(contract.N_STATS):
        i = contract.feature_index(2, st)
        assert feats[i] == contract.SENTINEL_MISSING, (
            f"{feature_names()[i]} is {feats[i]}, expected the sentinel")


def test_open_decisions_are_none_not_guessed():
    """
    Level 3. A guessed constant is worse than no constant, because a guess
    looks exactly like a decision six weeks later.
    """
    for name in ("DETECTION_THRESHOLD", "ALERT_HOLD_S"):
        v = getattr(contract, name)
        assert v is None or isinstance(v, (int, float)), (
            f"{name} is {v!r}; it should be None (undecided) or a number "
            f"(decided), never anything else")


def test_runtime_header_refuses_to_compile_while_undecided():
    """
    The #error is the mechanism, so verify it actually fires.

    Skipped if no C compiler is present, and skipped if every decision has
    since been made -- at which point the header SHOULD compile.
    """
    import shutil
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        print("    (skipped: no C compiler)")
        return

    undecided = [n for n in ("DETECTION_THRESHOLD", "ALERT_HOLD_S")
                 if getattr(contract, n) is None]
    if not undecided:
        print("    (skipped: every decision has been made)")
        return

    r = subprocess.run([cc, "-x", "c", "-fsyntax-only", "-"],
                       cwd=HERE, input='#include "contract_runtime.h"\n',
                       capture_output=True, text=True)
    assert r.returncode != 0, (
        f"contract_runtime.h compiled while {undecided} are still unset. "
        f"The #error is the mechanism; if it does not fire, the firmware "
        f"will happily build on a guess.")
    assert "unset" in r.stderr, (
        f"it failed to compile, but not with the contract's #error:\n"
        f"{r.stderr[:400]}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}\n       {e}")
        else:
            print(f"  ok   {t.__name__}")
    print(f"\n  {len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
