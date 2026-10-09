"""
features.py -- ONE HALF OF A SEAM.

The other half is features.c, and the two are pinned together by
golden_vectors.json. Neither file is the specification; the golden vectors
are. Both implementations have to reproduce them.

=============================================================================
WHAT IS ON THE SEAM AND WHY IT IS DANGEROUS
=============================================================================

This module turns a window of raw sensor samples into a fixed-length feature
vector. That is a FUNCTION, and a function is what a contract hash cannot
check. Four decisions inside it will silently diverge between two
implementations written months apart:

  1. SCALING. Raw ticks -> physical units. Swap the accelerometer and
     gyroscope factors and every feature is wrong by a constant ratio.
     Nothing crashes.

  2. THE SENTINEL. What counts as "no reading", and what happens to a window
     that contains some. Excluded from the statistics here. If one side
     includes them instead, it averages -32768 into a real measurement.

  3. ddof. numpy.std uses ddof=0. pandas.Series.std uses ddof=1. A naive C
     loop writes ddof=0. Pick pandas by accident in Python and every std
     feature is off by sqrt(n/(n-1)) -- about 0.5% at n=100 -- in Python
     only.

  4. LAYOUT. feature[ch * N_STATS + stat] against
     feature[stat * N_CHANNELS + ch]. Every value present, every value
     correct, every value in the wrong slot. The most invisible bug here.

All four are covered by golden vectors. None would be caught by a hash.

=============================================================================
CONVENTIONS THIS MODULE FIXES
=============================================================================

A window is (WINDOW_SAMPLES, N_CHANNELS) of raw int16 ticks, in contract
channel order. It is NOT scaled beforehand -- scaling happens here, once, so
there is exactly one place it can be wrong.

A fully-missing channel yields SENTINEL_MISSING for all four of its
statistics. Returning 0.0 would be indistinguishable from a real, valid
reading of a sensor at rest.
"""

from __future__ import annotations

import numpy as np

from contract import (
    CHANNELS, N_CHANNELS, STATS, N_STATS, N_FEATURES, WINDOW_SAMPLES,
    STD_DDOF, ACCEL_TICKS_PER_G, GYRO_TICKS_PER_DPS, SENTINEL_MISSING,
    feature_index,
)

# Per-channel scale factors, derived from the channel names rather than
# listed separately. A separate list would be a second place the mapping
# could be wrong.
SCALES = np.array(
    [1.0 / ACCEL_TICKS_PER_G if c.startswith("a") else 1.0 / GYRO_TICKS_PER_DPS
     for c in CHANNELS],
    dtype=np.float64,
)


def scale_window(raw):
    """
    Raw int16 ticks -> physical units, sentinels preserved.

    A sentinel is a marker, not a measurement, so it must not be multiplied
    by a scale factor. Scaling it would produce -8.0 g, which is a perfectly
    plausible reading and would then be silently treated as one.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (WINDOW_SAMPLES, N_CHANNELS):
        raise ValueError(
            f"expected a ({WINDOW_SAMPLES}, {N_CHANNELS}) window in contract "
            f"channel order {CHANNELS}, got {raw.shape}")

    missing = raw == SENTINEL_MISSING
    out = raw * SCALES
    out[missing] = SENTINEL_MISSING
    return out


def extract_features(raw):
    """
    (WINDOW_SAMPLES, N_CHANNELS) raw ticks -> (N_FEATURES,) float64.

    Returns the feature vector in the layout the contract declares. Never
    reorders, never renames -- feature_index() is the single definition of
    where a (channel, stat) pair lives, shared with the C side through the
    generated header.
    """
    scaled = scale_window(raw)
    feats = np.zeros(N_FEATURES, dtype=np.float64)

    for ch in range(N_CHANNELS):
        col = scaled[:, ch]
        valid = col[col != SENTINEL_MISSING]

        if valid.size == 0:
            # Every sample missing. Propagate the sentinel rather than
            # inventing a number: 0.0 here would read as a valid,
            # unremarkable measurement.
            for st in range(N_STATS):
                feats[feature_index(ch, st)] = SENTINEL_MISSING
            continue

        n = valid.size
        mean = valid.sum() / n

        # ddof is explicit. See the header note -- this is the one a naive C
        # loop writes, and the golden vectors lock it in both languages.
        denom = n - STD_DDOF
        std = np.sqrt(((valid - mean) ** 2).sum() / denom) if denom > 0 else 0.0

        feats[feature_index(ch, "mean")] = mean
        feats[feature_index(ch, "std")] = std
        feats[feature_index(ch, "min")] = valid.min()
        feats[feature_index(ch, "max")] = valid.max()

    return feats


def feature_names():
    """Human-readable name per index, in contract layout order."""
    names = [None] * N_FEATURES
    for ch, cname in enumerate(CHANNELS):
        for st, sname in enumerate(STATS):
            names[feature_index(ch, st)] = f"{cname}_{sname}"
    return names
