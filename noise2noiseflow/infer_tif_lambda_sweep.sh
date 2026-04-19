#!/usr/bin/env bash
#
# Run inference for each λ-specific checkpoint from run_lambda_sweep.sh.
#
# Prerequisite:
#   User has placed the final checkpoint for each λ as:
#     experiments/archive/n2nf_lmbda_{LMBDA}_5ms_last.pth
#
# Output directory (separate from the main data pipeline):
#   lambda_sweep_outputs/5ms/
#     test_denoised_lmbda{LMBDA}.tif
#     bg_denoised_lmbda{LMBDA}.tif
#
# Usage:
#   bash infer_tif_lambda_sweep.sh

set -e

TIME=5
DATA_DIR="../../data-prep/data/${TIME}ms_array_bs"
CKPT_DIR="./experiments/archive"
OUT_DIR="./lambda_sweep_outputs/${TIME}ms"

mkdir -p "$OUT_DIR"

# Must match run_lambda_sweep.sh
LMBDA_LIST=(65536 262144 1048576 16777216)

# 5ms parameters
VMIN=473
VMAX=647
BIAS=500.182
READOUT=9.019
CIC=0.01003

for LMBDA in "${LMBDA_LIST[@]}"; do
    CKPT="${CKPT_DIR}/n2nf_lmbda_${LMBDA}_${TIME}ms_last.pth"

    if [ ! -f "$CKPT" ]; then
        echo "[skip] λ=${LMBDA}: checkpoint not found at $CKPT"
        continue
    fi

    OUT_MAIN="${OUT_DIR}/test_denoised_lmbda${LMBDA}.tif"
    OUT_BG="${OUT_DIR}/bg_denoised_lmbda${LMBDA}.tif"

    echo "=========================================="
    echo "  λ = ${LMBDA}"
    echo "  ckpt   = $CKPT"
    echo "  main   = $OUT_MAIN"
    echo "  bg out = $OUT_BG"
    echo "=========================================="

    python infer_tif.py \
        --input "${DATA_DIR}/test_fp_filtered.tif" \
        --output "$OUT_MAIN" \
        --bg_input "${DATA_DIR}/raw/${TIME}ms_data_background.tif" \
        --bg_output "$OUT_BG" \
        --ckpt "$CKPT" \
        --vmin "$VMIN" \
        --vmax "$VMAX" \
        --basden_bias_offset "$BIAS" \
        --basden_readout_sigma "$READOUT" \
        --basden_cic_lambda "$CIC"
done

echo ""
echo "=========================================="
echo "  Inference sweep done."
echo "  Outputs in: $OUT_DIR"
ls -la "$OUT_DIR"/ 2>/dev/null || true
echo "=========================================="
