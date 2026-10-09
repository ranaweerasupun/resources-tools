/*
 * features.c -- ONE HALF OF A SEAM.
 *
 * The other half is features.py. Neither is the specification: golden_vectors.h
 * is. This file is not "the Python algorithm written again and trusted" -- it
 * is an implementation that must reproduce a fixed set of committed numbers or
 * the build fails.
 *
 * Written as plain C99 with no platform dependencies, so the same source that
 * would run on a microcontroller also compiles on a laptop with `cc`. That is
 * what makes the golden test runnable without hardware.
 *
 * Deliberately includes ONLY contract_data.h -- the seam. Not
 * contract_runtime.h, which #errors while product thresholds are still open.
 * Fusing the two would mean the seam could not be tested until every threshold
 * was frozen, which is backwards: the seam is what you want pinned first.
 */

#include "features.h"

#include <math.h>

/*
 * Per-channel scale factor, keyed off the channel index.
 *
 * The switch is over the generated CH_* constants rather than a hand-written
 * lookup table, so adding a channel to the contract produces a compile error
 * here rather than a silently wrong factor.
 */
static float channel_scale(int ch)
{
    switch (ch) {
    case CH_AX:
    case CH_AY:
    case CH_AZ:
        return 1.0f / ACCEL_TICKS_PER_G;
    case CH_GX:
    case CH_GY:
    case CH_GZ:
        return 1.0f / GYRO_TICKS_PER_DPS;
    default:
        return 0.0f;
    }
}

void features_extract(const int16_t raw[WINDOW_SAMPLES][N_CHANNELS],
                      float out[N_FEATURES])
{
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const float scale = channel_scale(ch);

        /* Pass 1: sum and extremes over the valid samples only.
         *
         * A sentinel is a marker, not a measurement. Scaling it would give
         * -8.0 g, a perfectly plausible reading that would then be treated
         * as real. So the comparison is against the RAW value, before any
         * multiplication. */
        int n = 0;
        double sum = 0.0;
        float vmin = 0.0f, vmax = 0.0f;

        for (int i = 0; i < WINDOW_SAMPLES; i++) {
            if ((float)raw[i][ch] == SENTINEL_MISSING) {
                continue;
            }
            const float v = (float)raw[i][ch] * scale;
            if (n == 0 || v < vmin) { vmin = v; }
            if (n == 0 || v > vmax) { vmax = v; }
            sum += (double)v;
            n++;
        }

        if (n == 0) {
            /* Every sample missing. Propagate the sentinel rather than
             * inventing a number -- 0.0 would read as a valid measurement of
             * a sensor sitting still. */
            for (int st = 0; st < N_STATS; st++) {
                out[FEATURE_INDEX(ch, st)] = SENTINEL_MISSING;
            }
            continue;
        }

        const double mean = sum / (double)n;

        /* Pass 2: variance about that mean.
         *
         * STD_DDOF comes from the contract. numpy.std defaults to 0 and
         * pandas.Series.std defaults to 1; they differ by sqrt(n/(n-1)),
         * which is roughly 0.5% at n=100. Nothing crashes if the two sides
         * disagree -- the features are just quietly wrong. Hence the golden
         * vectors. */
        double ss = 0.0;
        for (int i = 0; i < WINDOW_SAMPLES; i++) {
            if ((float)raw[i][ch] == SENTINEL_MISSING) {
                continue;
            }
            const double d = (double)((float)raw[i][ch] * scale) - mean;
            ss += d * d;
        }

        const int denom = n - STD_DDOF;
        const double var = (denom > 0) ? (ss / (double)denom) : 0.0;

        out[FEATURE_INDEX(ch, STAT_MEAN)] = (float)mean;
        out[FEATURE_INDEX(ch, STAT_STD)]  = (float)sqrt(var);
        out[FEATURE_INDEX(ch, STAT_MIN)]  = vmin;
        out[FEATURE_INDEX(ch, STAT_MAX)]  = vmax;
    }
}
