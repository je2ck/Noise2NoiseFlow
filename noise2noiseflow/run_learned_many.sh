#!/bin/bash
#
# Pure learned-flow baseline (no Basden physics layer).
# Mirrors run_basden_many.sh but swaps arch to sq|unc|unc|gain|unc|unc|usq.
# Used to quantify how much of the performance is due to physics injection
# vs pure data-driven flow learning.
#

set -e

TIMES=("5ms" "4ms" "8ms" "10ms" "12ms" "14ms" "16ms" "20ms")

DATA_ROOT="./data_atom"
LOG_ROOT="experiments/paper"

for TIME in "${TIMES[@]}"
do
    case "$TIME" in
        "20ms") CURRENT_VMIN=473; CURRENT_VMAX=805 ;;
        "16ms") CURRENT_VMIN=473; CURRENT_VMAX=769 ;;
        "14ms") CURRENT_VMIN=473; CURRENT_VMAX=754 ;;
        "12ms") CURRENT_VMIN=473; CURRENT_VMAX=731 ;;
        "10ms") CURRENT_VMIN=473; CURRENT_VMAX=702 ;;
        "8ms")  CURRENT_VMIN=473; CURRENT_VMAX=678 ;;
        "5ms")  CURRENT_VMIN=473; CURRENT_VMAX=647 ;;
        "4ms")  CURRENT_VMIN=473; CURRENT_VMAX=654 ;;
        *) echo "Error: undefined TIME -> $TIME"; exit 1 ;;
    esac

    echo ""
    echo "=========================================="
    echo "Starting LEARNED-flow training for: $TIME"
    echo "Norm: VMIN=$CURRENT_VMIN, VMAX=$CURRENT_VMAX"
    echo "Arch: sq|unc|unc|gain|unc|unc|gain|unc|unc|usq  (no Basden physics)"
    echo "=========================================="

    LOG_DIR_NAME="n2nf_learned_only_${TIME}"

    CURRENT_DATA_PATH="${DATA_ROOT}/${TIME}"
    if [ ! -d "$CURRENT_DATA_PATH" ]; then
        echo "Error: Data directory not found -> $CURRENT_DATA_PATH"
        exit 1
    fi

    MIN_EPOCHS=200
    MAX_EPOCHS=500
    EARLY_STOP_PATIENCE=5
    EARLY_STOP_MIN_DELTA=0.0005

    echo ">>> [${TIME}] arch=sq|unc|unc|gain|unc|unc|gain|unc|unc|usq, epochs=${MIN_EPOCHS}..${MAX_EPOCHS}"
    python train_atom.py \
        --arch "sq|unc|unc|gain|unc|unc|gain|unc|unc|usq" \
        --epochs "$MAX_EPOCHS" \
        --logdir "$LOG_DIR_NAME" \
        --sidd_path "$CURRENT_DATA_PATH" \
        --n_batch_train 8 \
        --n_batch_test 8 \
        --n_patches_per_image 1 \
        --patch_height 64 \
        --patch_sampling uniform \
        --n_channels 1 \
        --epochs_full_valid 10 \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
        --early_stop_min_epoch "$MIN_EPOCHS" \
        --lu_decomp \
        --lmbda 262144 \
        --dncnn_num_layers 25 \
        --no_resume \
        --n_train_threads 0 \
        --vmin "$CURRENT_VMIN" \
        --vmax "$CURRENT_VMAX"

    sleep 60

    echo ">>> Checkpoints and logs saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
done

echo ""
echo "=========================================="
echo "All LEARNED-flow training sessions completed."
echo "=========================================="
