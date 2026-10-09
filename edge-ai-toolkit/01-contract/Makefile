# Host build of the golden test. Plain cc, no board, no SDK.
#
#     make run     build and run the C golden test
#     make all     the full three-level check, as CI would run it
#     make clean

CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -O2
LDLIBS  := -lm

BIN     := test_golden
SRC     := test_golden.c features.c
GEN     := contract.py contract_data.h contract_runtime.h \
           golden_vectors.json golden_vectors.h

.PHONY: all run check clean generate

# The order matters and it is a dependency chain, not a preference:
# gen_golden.py imports what data_contract.py generates, and stamps the
# contract hash into both golden files. Regenerate the contract without
# regenerating the vectors and the C test fails on the hash comparison --
# which is the intended behaviour, not a nuisance.
generate:
	python3 data_contract.py
	python3 gen_golden.py

$(BIN): $(SRC) features.h contract_data.h golden_vectors.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

run: $(BIN)
	./$(BIN)

# What CI runs. Each level catches what the one below it cannot.
check:
	@echo "== level 1: constants ======================================"
	python3 data_contract.py --check
	@echo
	@echo "== level 2: functions ======================================"
	python3 gen_golden.py --check
	$(MAKE) --no-print-directory run
	@echo
	@echo "== python side ============================================="
	python3 test_contract.py

clean:
	rm -f $(BIN)

# Deliberately absent: any rule that builds contract_runtime.h into
# something. It #errors while DETECTION_THRESHOLD and ALERT_HOLD_S are
# unset, and that is the point -- the seam is testable before the product
# thresholds are frozen. Try it:
#
#     echo '#include "contract_runtime.h"' | cc -x c -c - -o /dev/null
