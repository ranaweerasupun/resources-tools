"""
synth.py -- deterministic synthetic sensor dataset.

Shared by every benchmark in this series so that results are comparable and
reproducible on any machine, with no hardware and no downloaded dataset.

WHY THE DATA IS SHAPED THIS WAY
-------------------------------
Compression results are meaningless if the test data is wrong in character.
np.random.randn() produces float32 values that use the full mantissa, and
nothing compresses it -- you would measure Parquet at ~1.0x and conclude
columnar storage does not help. Real sensor data is not like that:

  * It comes from a fixed-point ADC. An IMU gives you int16 ticks which you
    scale to physical units, so the underlying values are quantised, not
    continuous. Only a few thousand distinct values ever appear.
  * It is autocorrelated. Sample n+1 is close to sample n. Consecutive values
    share high bits, which is what byte-stream compressors exploit.
  * Its metadata columns are extremely repetitive. A session id repeats for
    every row of that session. That is what dictionary and run-length
    encoding are for, and it is where columnar formats win hardest.
  * Its timestamps are monotonic with a near-constant step, which
    delta-encodes almost to nothing.

So this generator produces quantised, autocorrelated channels with realistic
metadata. The numbers you measure with it transfer to real data; numbers
measured on random noise do not.

USAGE
-----
    from synth import generate
    df = generate(sessions=200, seconds=120, rate_hz=50)

    python synth.py --sessions 200 --seconds 120 --out /tmp/data.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# --- schema ---------------------------------------------------------------
# Six IMU-like channels. Order is fixed and load-bearing: this mirrors the
# "channel order is part of the contract" point from post 1, and lets the
# same dataset be reused there.
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]

# Fixed-point scales, chosen to match common IMU configurations.
# +/-8 g over int16 -> 4096 ticks per g;  +/-2000 dps -> 16.384 ticks per dps.
ACCEL_TICKS_PER_G = 4096.0
GYRO_TICKS_PER_DPS = 16.384

# The dtypes the canonical table declares. Every format in the benchmark is
# asked to give these back unchanged; several cannot.
DTYPES = {
    "t_ms": "int64",
    "session": "int16",
    "subject": "int16",
    "label": "int8",
    "ax": "float32", "ay": "float32", "az": "float32",
    "gx": "float32", "gy": "float32", "gz": "float32",
}
COLUMNS = list(DTYPES)

SEED = 42


def _walk(rng, n, step, clip):
    """Autocorrelated series: a bounded random walk, like a real signal."""
    x = np.cumsum(rng.normal(0.0, step, n))
    return np.clip(x, -clip, clip)


def _session(rng, n, rate_hz, moving):
    """
    One session's six channels, as int16 ticks.

    Returns ticks rather than physical units because that is what the sensor
    actually produces. Quantisation is the point: it is why the data
    compresses at all.
    """
    # Accelerometer: gravity on one axis plus motion. Orientation is random
    # per session, which is what a wrist or a handheld device really does.
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, np.pi)
    gravity = np.array([np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi)])

    motion = 0.30 if moving else 0.02      # g, rms
    spin = 60.0 if moving else 1.0         # dps

    acc = np.column_stack([
        gravity[i] + _walk(rng, n, motion / np.sqrt(rate_hz), 4.0)
        for i in range(3)
    ])
    gyr = np.column_stack([
        _walk(rng, n, spin / np.sqrt(rate_hz), 1000.0)
        for _ in range(3)
    ])

    # Quantise through the ADC, exactly as the hardware would.
    acc_ticks = np.clip(np.round(acc * ACCEL_TICKS_PER_G), -32768, 32767)
    gyr_ticks = np.clip(np.round(gyr * GYRO_TICKS_PER_DPS), -32768, 32767)
    return acc_ticks.astype(np.int16), gyr_ticks.astype(np.int16)


def generate(sessions=200, seconds=120, rate_hz=50, subjects=12, seed=SEED,
             scaled=True):
    """
    Build the canonical table.

    sessions : number of recordings
    seconds  : length of each recording
    rate_hz  : sample rate
    subjects : how many distinct people the sessions are spread over
    scaled   : True  -> channels in physical units as float32 (the canonical
                        form most pipelines use)
               False -> channels as raw int16 ticks (what the sensor gave you)

    Row count is sessions * seconds * rate_hz.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * rate_hz)
    step_ms = int(round(1000.0 / rate_hz))

    frames = []
    for s in range(sessions):
        moving = bool(rng.random() < 0.5)
        acc_ticks, gyr_ticks = _session(rng, n, rate_hz, moving)

        cols = {
            # Monotonic within a session, restarting each time -- delta
            # encoding handles this well, plain storage does not.
            "t_ms": np.arange(n, dtype=np.int64) * step_ms,
            "session": np.full(n, s, dtype=np.int16),
            "subject": np.full(n, s % subjects, dtype=np.int16),
            "label": np.full(n, 1 if moving else 0, dtype=np.int8),
        }
        ticks = np.concatenate([acc_ticks, gyr_ticks], axis=1)
        if scaled:
            scales = np.array([1 / ACCEL_TICKS_PER_G] * 3 +
                              [1 / GYRO_TICKS_PER_DPS] * 3, dtype=np.float32)
            phys = (ticks.astype(np.float32) * scales)
            for i, ch in enumerate(CHANNELS):
                cols[ch] = phys[:, i]
        else:
            for i, ch in enumerate(CHANNELS):
                cols[ch] = ticks[:, i]

        frames.append(pd.DataFrame(cols))

    df = pd.concat(frames, ignore_index=True)[COLUMNS]

    # Assert the dtypes we claim to produce. If this fails, every downstream
    # fidelity result is measuring the generator's bug, not the format's.
    want = dict(DTYPES)
    if not scaled:
        want.update({c: "int16" for c in CHANNELS})
    for col, dt in want.items():
        if str(df[col].dtype) != dt:
            raise AssertionError(
                f"generator produced {col} as {df[col].dtype}, expected {dt}")
    return df


def describe(df):
    """Facts worth printing next to any benchmark result."""
    mem = int(df.memory_usage(deep=True).sum())
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "sessions": int(df["session"].nunique()),
        "subjects": int(df["subject"].nunique()),
        "in_memory_bytes": mem,
        "in_memory_mib": round(mem / 1024**2, 1),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--rate-hz", type=float, default=50)
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--raw-ticks", action="store_true",
                    help="emit int16 ticks instead of scaled float32")
    ap.add_argument("--out", help="write to this .parquet path")
    args = ap.parse_args()

    df = generate(args.sessions, args.seconds, args.rate_hz, args.subjects,
                  scaled=not args.raw_ticks)
    info = describe(df)
    print(f"  {info['rows']:,} rows x {info['columns']} cols   "
          f"{info['sessions']} sessions   {info['subjects']} subjects")
    print(f"  {info['in_memory_mib']} MiB in memory")
    for c, t in info["dtypes"].items():
        print(f"    {c:<10} {t}")

    if args.out:
        df.to_parquet(args.out, index=False)
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
