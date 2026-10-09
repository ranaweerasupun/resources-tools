/*
 * features.h -- the C half of the seam.
 *
 * Includes ONLY contract_data.h. contract_runtime.h is deliberately absent:
 * it #errors while product thresholds remain undecided, and the seam must be
 * testable before those are frozen.
 */

#ifndef FEATURES_H
#define FEATURES_H

#include <stdint.h>

#include "contract_data.h"

/*
 * One window of raw sensor ticks -> the feature vector.
 *
 * raw : WINDOW_SAMPLES x N_CHANNELS of signed 16-bit ADC ticks, in contract
 *       channel order (CH_AX .. CH_GZ). NOT pre-scaled; scaling happens
 *       inside, once, so there is exactly one place it can be wrong.
 *       Samples equal to SENTINEL_MISSING are excluded from every statistic.
 *
 * out : N_FEATURES floats, written in the contract's layout. Use
 *       FEATURE_INDEX(ch, stat) to read one back -- never hand-roll the
 *       arithmetic, or this file and the Python side can disagree about
 *       where a value lives while both look correct.
 *
 * A channel whose samples are all missing yields SENTINEL_MISSING for all
 * of its statistics.
 */
void features_extract(const int16_t raw[WINDOW_SAMPLES][N_CHANNELS],
                      float out[N_FEATURES]);

#endif /* FEATURES_H */
