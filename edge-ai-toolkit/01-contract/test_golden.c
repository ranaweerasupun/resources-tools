/*
 * test_golden.c -- make the C implementation prove it agrees with Python.
 *
 *     make run
 *
 * Plain cc, no ESP-IDF, no Zephyr, no board, about a second. This is the
 * part of an edge AI codebase that can gate a pull request without hardware,
 * and it is worth protecting: if features.c ever grows a dependency on a
 * peripheral, that capability is gone.
 *
 * A test suite that has never failed is decoration. Run sabotage.py to watch
 * this one fail on purpose, ten different ways.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "features.h"
#include "golden_vectors.h"

/*
 * Tolerance-based comparison, not exact equality.
 *
 * Python computes in float64, the device in float32. They will never match
 * to the last bit. A golden test demanding exact equality fails forever and
 * gets switched off within a week -- and a switched-off test is worse than
 * no test, because people believe it is running.
 *
 * The tolerances come from the contract, so both languages use the same
 * ones. Loose enough for float32 rounding; tight enough that a wrong
 * sentinel (off by thousands), a transposed layout (off by orders of
 * magnitude) or a ddof mismatch (off by ~0.5%) all fall well outside.
 */
static int close_enough(float got, float want)
{
    const float diff = fabsf(got - want);
    if (diff <= GOLDEN_ABS_TOL) {
        return 1;
    }
    const float scale = fabsf(want);
    return diff <= GOLDEN_REL_TOL * scale;
}

int main(void)
{
    /*
     * Before comparing anything: were these vectors generated against the
     * contract this binary was compiled against?
     *
     * Golden vectors certifying agreement on a seam that has since moved
     * are worse than none, because they report success.
     */
    if (strcmp(CONTRACT_HASH, GOLDEN_CONTRACT_HASH) != 0) {
        printf("CONTRACT HASH MISMATCH\n");
        printf("  contract_data.h    %s\n", CONTRACT_HASH);
        printf("  golden_vectors.h   %s\n", GOLDEN_CONTRACT_HASH);
        printf("\n  The golden vectors were made from a different version of the contract.\n");
        printf("  Run: python data_contract.py && python gen_golden.py\n");
        return 2;
    }

    int failures = 0;
    int comparisons = 0;

    for (int s = 0; s < N_GOLDEN_SCENES; s++) {
        const golden_scene_t *sc = &GOLDEN_SCENES[s];
        float got[N_FEATURES];

        /* Fill every slot with NaN first. A slot that features_extract()
         * never writes then fails on every platform, instead of passing or
         * failing depending on whatever was left on the stack. */
        for (int f = 0; f < N_FEATURES; f++) {
            got[f] = NAN;
        }

        features_extract(sc->input, got);

        int scene_failures = 0;
        for (int f = 0; f < N_FEATURES; f++) {
            comparisons++;
            if (!close_enough(got[f], sc->expected[f])) {
                if (scene_failures == 0) {
                    printf("FAIL %-18s (%s)\n", sc->name, sc->catches);
                }
                printf("       %-10s [%2d]  C %-14.6f  Python %-14.6f  "
                       "diff %.6g\n",
                       GOLDEN_FEATURE_NAMES[f], f,
                       (double)got[f], (double)sc->expected[f],
                       (double)fabsf(got[f] - sc->expected[f]));
                scene_failures++;
                failures++;
            }
        }
        if (scene_failures == 0) {
            printf("ok   %-18s %s\n", sc->name, sc->catches);
        }
    }

    printf("\n%d scenes, %d feature comparisons, %d failures\n",
           N_GOLDEN_SCENES, comparisons, failures);

    if (failures) {
        printf("\nThe C implementation does not agree with the Python "
               "reference.\n");
        printf("The golden vectors are the specification; both sides must "
               "match them.\n");
        return 1;
    }
    return 0;
}
