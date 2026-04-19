#!/usr/bin/env bash
#
# Fast fidelity eval for DnCNN depth sweep.

set -e

TIME=5
DATA_DIR="../../data-prep/data/${TIME}ms_array_bs"
DATA_PREP_DIR="../../data-prep"
OUT_DIR="./depth_sweep_outputs/${TIME}ms"
OUTPUT_LOG="${OUT_DIR}/fidelity_sweep_fast.txt"

DEPTH_LIST=(9 17 25)

MAIN_TIF="${DATA_DIR}/test_denoised.tif"
BG_TIF="${DATA_DIR}/bg_denoised.tif"
BACKUP_SUFFIX=".depth_backup"

[ -f "$MAIN_TIF" ] && cp "$MAIN_TIF" "${MAIN_TIF}${BACKUP_SUFFIX}"
[ -f "$BG_TIF" ]   && cp "$BG_TIF"   "${BG_TIF}${BACKUP_SUFFIX}"

restore() {
    [ -f "${MAIN_TIF}${BACKUP_SUFFIX}" ] && mv "${MAIN_TIF}${BACKUP_SUFFIX}" "$MAIN_TIF"
    [ -f "${BG_TIF}${BACKUP_SUFFIX}" ]   && mv "${BG_TIF}${BACKUP_SUFFIX}"   "$BG_TIF"
    echo "Restored."
}
trap restore EXIT

: > "$OUTPUT_LOG"
echo "DnCNN depth sweep fidelity (${TIME}ms)" >> "$OUTPUT_LOG"
echo "==========================================" >> "$OUTPUT_LOG"

for DEPTH in "${DEPTH_LIST[@]}"; do
    SRC_MAIN="${OUT_DIR}/test_denoised_depth${DEPTH}.tif"
    SRC_BG="${OUT_DIR}/bg_denoised_depth${DEPTH}.tif"

    if [ ! -f "$SRC_MAIN" ] || [ ! -f "$SRC_BG" ]; then
        echo "[skip] depth=${DEPTH}: missing outputs"
        echo "[skip] depth=${DEPTH}" >> "$OUTPUT_LOG"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  DnCNN depth = ${DEPTH}"
    echo "=========================================="

    cp "$SRC_MAIN" "$MAIN_TIF"
    cp "$SRC_BG"   "$BG_TIF"

    echo "" >> "$OUTPUT_LOG"
    echo "=========================================" >> "$OUTPUT_LOG"
    echo "  DnCNN depth = ${DEPTH}" >> "$OUTPUT_LOG"
    echo "=========================================" >> "$OUTPUT_LOG"
    (cd "$DATA_PREP_DIR" && python make_fidelity_table.py \
        --n-boot 5 \
        --n-folds 3 \
        --exposures "$TIME" \
        --skip raw_1d raw_2d den_3d bayes_md \
        --skip-plots) 2>&1 \
        | tee -a "$OUTPUT_LOG"
done

echo ""
echo "=========================================="
echo "  Done → $OUTPUT_LOG"
echo "=========================================="
