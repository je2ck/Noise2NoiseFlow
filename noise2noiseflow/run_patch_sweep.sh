#!/bin/bash
#
# Patch size / augmentation sweep on 5ms. Tests whether smaller
# random-crop patches + more patches per image (spatial augmentation)
# improves classification fidelity.
#
# Output:
#   experiments/paper/n2nf_patch_{PATCH}_ppi{PPI}_5ms/
#
# After training, place last-epoch ckpt under:
#   experiments/archive/n2nf_patch_{PATCH}_ppi{PPI}_5ms_last.pth

set -e

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
# (patch_size, n_patches_per_image) pairs to sweep
# ==========================
# (64, 1) : current default — patch == full image (no real augmentation)
# (48, 2) : mild augmentation, less boundary effect
# (32, 4) : Option A — aggressive augmentation, ~4× samples per epoch
# Adjust CONFIG_LIST below.
CONFIG_LIST=(
    "64 1"
    "48 2"
    "32 4"
)

EPOCHS=250

for CFG in "${CONFIG_LIST[@]}"; do
    read PATCH PPI <<< "$CFG"
    LOG_DIR_NAME="n2nf_patch_${PATCH}_ppi${PPI}_${TIME}"

    echo ""
    echo "=========================================="
    echo "  patch=${PATCH}  ppi=${PPI}   (time=${TIME})"
    echo "  logdir = ${LOG_DIR_NAME}"
    echo "=========================================="

    python train_atom.py \
        --arch "basden" \
        --epochs "$EPOCHS" \
        --logdir "$LOG_DIR_NAME" \
        --sidd_path "$CURRENT_DATA_PATH" \
        --n_batch_train 8 \
        --n_batch_test 8 \
        --n_patches_per_image "$PPI" \
        --patch_height "$PATCH" \
        --patch_sampling uniform \
        --n_channels 1 \
        --epochs_full_valid 10 \
        --lu_decomp \
        --lmbda 262144 \
        --dncnn_num_layers 25 \
        --no_resume \
        --n_train_threads 0 \
        --vmin "$CURRENT_VMIN" \
        --vmax "$CURRENT_VMAX" \
        --basden_bias_offset "$CURRENT_BIAS" \
        --basden_readout_sigma "$CURRENT_SIGMA" \
        --basden_em_gain "$CURRENT_GAIN" \
        --basden_cic_lambda "$CURRENT_CIC"

    echo ">>> [${PATCH}/${PPI}] saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
    sleep 30
done

echo ""
echo "=========================================="
echo "  Patch sweep complete."
echo "  Checkpoints:"
for CFG in "${CONFIG_LIST[@]}"; do
    read PATCH PPI <<< "$CFG"
    echo "    ${LOG_ROOT}/n2nf_patch_${PATCH}_ppi${PPI}_${TIME}/"
done
echo "=========================================="
