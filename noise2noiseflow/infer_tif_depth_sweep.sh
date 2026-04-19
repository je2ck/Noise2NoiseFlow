#!/usr/bin/env bash
#
# Infer for each DnCNN depth-specific checkpoint.
# Expects: experiments/archive/n2nf_depth_{D}_5ms_last.pth
# Outputs to: depth_sweep_outputs/5ms/

set -e

TIME=5
DATA_DIR="../../data-prep/data/${TIME}ms_array_bs"
CKPT_DIR="./experiments/archive"
OUT_DIR="./depth_sweep_outputs/${TIME}ms"
mkdir -p "$OUT_DIR"

DEPTH_LIST=(9 17 25)

VMIN=473
VMAX=647
BIAS=500.182
READOUT=9.019
CIC=0.01003

for DEPTH in "${DEPTH_LIST[@]}"; do
    CKPT="${CKPT_DIR}/n2nf_depth_${DEPTH}_${TIME}ms_last.pth"
    if [ ! -f "$CKPT" ]; then
        echo "[skip] depth=${DEPTH}: $CKPT not found"
        continue
    fi

    OUT_MAIN="${OUT_DIR}/test_denoised_depth${DEPTH}.tif"
    OUT_BG="${OUT_DIR}/bg_denoised_depth${DEPTH}.tif"

    echo "=========================================="
    echo "  DnCNN depth = ${DEPTH}"
    echo "  ckpt = $CKPT"
    echo "  out  = $OUT_MAIN"
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
        --basden_cic_lambda "$CIC" \
        --dncnn_num_layers "$DEPTH"
done

echo ""
echo "=========================================="
echo "  Outputs in: $OUT_DIR"
ls -la "$OUT_DIR"/ 2>/dev/null || true
echo "=========================================="
