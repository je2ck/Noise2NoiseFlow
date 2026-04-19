#!/usr/bin/env bash
#
# Evaluate fidelity for each λ-specific denoised output produced by
# infer_tif_lambda_sweep.sh.
#
# Reads λ-tagged denoised TIFs from:
#   lambda_sweep_outputs/5ms/test_denoised_lmbda{LMBDA}.tif
#   lambda_sweep_outputs/5ms/bg_denoised_lmbda{LMBDA}.tif
#
# make_fidelity_table.py hard-codes the denoised TIF filename via
# ExposureConfig (test_denoised.tif / bg_denoised.tif in the data dir).
# We temporarily swap in each λ-tagged file, run the fidelity table,
# then restore on exit.
#
# Output:
#   lambda_sweep_outputs/5ms/fidelity_sweep.txt
#     (make_fidelity_table stdout per λ, concatenated)
#
# Usage:
#   bash eval_fidelity_lambda_sweep.sh

set -e

TIME=5
DATA_DIR="../../data-prep/data/${TIME}ms_array_bs"
DATA_PREP_DIR="../../data-prep"
OUT_DIR="./lambda_sweep_outputs/${TIME}ms"
OUTPUT_LOG="${OUT_DIR}/fidelity_sweep.txt"

LMBDA_LIST=(65536 262144 1048576 16777216)

MAIN_TIF="${DATA_DIR}/test_denoised.tif"
BG_TIF="${DATA_DIR}/bg_denoised.tif"
BACKUP_SUFFIX=".sweep_backup"

# Back up existing default-named outputs (if any)
if [ -f "$MAIN_TIF" ]; then
    cp "$MAIN_TIF" "${MAIN_TIF}${BACKUP_SUFFIX}"
    echo "Backed up ${MAIN_TIF}"
fi
if [ -f "$BG_TIF" ]; then
    cp "$BG_TIF" "${BG_TIF}${BACKUP_SUFFIX}"
    echo "Backed up ${BG_TIF}"
fi

restore() {
    if [ -f "${MAIN_TIF}${BACKUP_SUFFIX}" ]; then
        mv "${MAIN_TIF}${BACKUP_SUFFIX}" "$MAIN_TIF"
        echo "Restored ${MAIN_TIF}"
    fi
    if [ -f "${BG_TIF}${BACKUP_SUFFIX}" ]; then
        mv "${BG_TIF}${BACKUP_SUFFIX}" "$BG_TIF"
        echo "Restored ${BG_TIF}"
    fi
}
trap restore EXIT

: > "$OUTPUT_LOG"
echo "Lambda sweep fidelity results (${TIME}ms)" >> "$OUTPUT_LOG"
echo "==========================================" >> "$OUTPUT_LOG"

for LMBDA in "${LMBDA_LIST[@]}"; do
    SRC_MAIN="${OUT_DIR}/test_denoised_lmbda${LMBDA}.tif"
    SRC_BG="${OUT_DIR}/bg_denoised_lmbda${LMBDA}.tif"

    if [ ! -f "$SRC_MAIN" ] || [ ! -f "$SRC_BG" ]; then
        echo "[skip] λ=${LMBDA}: missing ${SRC_MAIN} or ${SRC_BG}"
        echo "" >> "$OUTPUT_LOG"
        echo "[skip] λ=${LMBDA}: missing denoised outputs" >> "$OUTPUT_LOG"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  λ = ${LMBDA}"
    echo "=========================================="

    cp "$SRC_MAIN" "$MAIN_TIF"
    cp "$SRC_BG"   "$BG_TIF"

    echo "" >> "$OUTPUT_LOG"
    echo "=========================================" >> "$OUTPUT_LOG"
    echo "  λ = ${LMBDA}" >> "$OUTPUT_LOG"
    echo "=========================================" >> "$OUTPUT_LOG"
    (cd "$DATA_PREP_DIR" && python make_fidelity_table.py \
        --n-boot 50 \
        --exposures "$TIME") 2>&1 \
        | tee -a "$OUTPUT_LOG"
done

echo ""
echo "=========================================="
echo "  All λ evaluated. Log → $OUTPUT_LOG"
echo "=========================================="
