/*
 * GENERATED FILE -- DO NOT EDIT
 * Source: data_contract.py   Contract: DC-001 r1.0   Hash: d5a07fe0a05f
 *
 * Editing this by hand puts it out of sync with its
 * counterpart in the other language, which is exactly the
 * failure this file exists to prevent. Edit data_contract.py
 * and regenerate ALL of them.
 */

#ifndef CONTRACT_DATA_H
#define CONTRACT_DATA_H

#define CONTRACT_HASH "d5a07fe0a05f"

/* --- sampling --- */
#define SAMPLE_RATE_HZ      50
#define WINDOW_MS           2000
#define WINDOW_SAMPLES      100

/* --- channels: ORDER IS LOAD-BEARING. The buffer is filled
   in this order and the model was trained on this order. --- */
#define N_CHANNELS          6
#define CH_AX               0
#define CH_AY               1
#define CH_AZ               2
#define CH_GX               3
#define CH_GY               4
#define CH_GZ               5

static const char *const CHANNEL_NAMES[N_CHANNELS] = { "ax", "ay", "az", "gx", "gy", "gz" };

/* --- feature layout --- */
#define N_STATS             4
#define N_FEATURES          24
#define STAT_MEAN           0
#define STAT_STD            1
#define STAT_MIN            2
#define STAT_MAX            3
#define STD_DDOF            0

/* layout: channel_major */
#define FEATURE_INDEX(ch, stat) ((ch) * N_STATS + (stat))

/* --- units --- */
#define ACCEL_TICKS_PER_G   4096.0f
#define GYRO_TICKS_PER_DPS  16.384f

/* --- sentinel: a value the sensor cannot produce.
   NOT 0.0 -- zero is what a zeroed struct gives you for free,
   and it is also a valid reading of a sensor at rest. --- */
#define SENTINEL_MISSING    -32768.0f

/* --- golden vector tolerances --- */
#define GOLDEN_REL_TOL      0.0001f
#define GOLDEN_ABS_TOL      0.001f

#endif /* CONTRACT_DATA_H */
