#!/bin/bash
#
# DnCNN depth sweep on 5ms. Tests whether larger DnCNN improves
# classification fidelity (saturation test).
#
# Output:
#   experiments/paper/n2nf_depth_{D}_5ms/  (one per depth)
#
# After training, place last-epoch ckpt under:
#   experiments/archive/n2nf_depth_{D}_5ms_last.pth
# and run infer_tif_depth_sweep.sh + eval.

set -e

# ==========================
# 5ms exposure parameters
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
# DnCNN depths to sweep
# ==========================
# 9  = current default
# 17 = standard DnCNN paper depth
# 25 = deeper
DEPTH_LIST=(9 17 25)

EPOCHS=250

for DEPTH in "${DEPTH_LIST[@]}"; do
    LOG_DIR_NAME="n2nf_depth_${DEPTH}_${TIME}"

    echo ""
    echo "=========================================="
    echo "  DnCNN depth = ${DEPTH}   (time = ${TIME})"
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
        --lmbda 262144 \
        --dncnn_num_layers "$DEPTH" \
        --no_resume \
        --n_train_threads 0 \
        --vmin "$CURRENT_VMIN" \
        --vmax "$CURRENT_VMAX" \
        --basden_bias_offset "$CURRENT_BIAS" \
        --basden_readout_sigma "$CURRENT_SIGMA" \
        --basden_em_gain "$CURRENT_GAIN" \
        --basden_cic_lambda "$CURRENT_CIC"

    echo ">>> [depth=${DEPTH}] saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
    sleep 30
done

echo ""
echo "=========================================="
echo "  DnCNN depth sweep complete."
echo "  Checkpoints:"
for DEPTH in "${DEPTH_LIST[@]}"; do
    echo "    ${LOG_ROOT}/n2nf_depth_${DEPTH}_${TIME}/"
done
echo "=========================================="
