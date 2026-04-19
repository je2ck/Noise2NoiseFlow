#!/bin/bash
#
# Lambda (MSE weight) sweep experiment on 5ms only.
# Goal: determine how lmbda affects classification fidelity — i.e., whether
#       the implicit bias from basden's NLL term helps or hurts the
#       downstream classifier features.
#
# Output:
#   experiments/paper/n2nf_lmbda_{LMBDA}_5ms/   (one per λ value)
#
# After training, run:
#   bash infer_tif_lambda_sweep.sh           # if you make one
#   (or manually point infer_tif.py at each ckpt and compare fidelity)

set -e

# ==========================
# 5ms exposure parameters (fixed)
# ==========================
TIME="5ms"
CURRENT_VMIN=473
CURRENT_VMAX=647
CURRENT_BIAS=500.182
CURRENT_SIGMA=9.019
CURRENT_GAIN=300.00
CURRENT_CIC=0.01003

DATA_ROOT="./data_atom"
CURRENT_DATA_PATH="${DATA_ROOT}/${TIME}"
LOG_ROOT="experiments/paper"

if [ ! -d "$CURRENT_DATA_PATH" ]; then
    echo "Error: Data directory not found -> $CURRENT_DATA_PATH"
    exit 1
fi

# ==========================
# λ values to sweep
# ==========================
# Current default = 262144 (≈ 2.6e5)
#   - 65536   : ×0.25 (NLL 영향 ↑, basden bias 더 강하게)
#   - 262144  : baseline
#   - 1048576 : ×4    (NLL 영향 ↓, MSE 비중 ↑)
#   - 16777216: ×64   (거의 pure MSE, NLL 영향 미미)
LMBDA_LIST=(65536 262144 1048576 16777216)

EPOCHS=250

# ==========================
# Loop
# ==========================
for LMBDA in "${LMBDA_LIST[@]}"; do
    LOG_DIR_NAME="n2nf_lmbda_${LMBDA}_${TIME}"

    echo ""
    echo "=========================================="
    echo "  λ = ${LMBDA}   (time = ${TIME})"
    echo "  logdir = ${LOG_DIR_NAME}"
    echo "=========================================="

    python train_atom.py \
        --arch "basden" \
        --epochs "$EPOCHS" \
        --logdir "$LOG_DIR_NAME" \
        --sidd_path "$CURRENT_DATA_PATH" \
        --n_batch_train 8 \
        --n_batch_test 8 \
        --n_patches_per_image 1 \
        --patch_height 64 \
        --patch_sampling uniform \
        --n_channels 1 \
        --epochs_full_valid 10 \
        --lu_decomp \
        --lmbda "$LMBDA" \
        --no_resume \
        --n_train_threads 0 \
        --vmin "$CURRENT_VMIN" \
        --vmax "$CURRENT_VMAX" \
        --basden_bias_offset "$CURRENT_BIAS" \
        --basden_readout_sigma "$CURRENT_SIGMA" \
        --basden_em_gain "$CURRENT_GAIN" \
        --basden_cic_lambda "$CURRENT_CIC"

    echo ">>> [λ=${LMBDA}] saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
    sleep 30
done

echo ""
echo "=========================================="
echo "  Lambda sweep complete."
echo "  Checkpoints:"
for LMBDA in "${LMBDA_LIST[@]}"; do
    echo "    ${LOG_ROOT}/n2nf_lmbda_${LMBDA}_${TIME}/"
done
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run inference on each ckpt (edit infer_tif.sh ckpt path or loop)"
echo "  2. Run fidelity_table on each to compare Den_1D / PSF_3D"
echo "=========================================="
