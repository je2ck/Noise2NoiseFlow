#!/bin/bash
#
# Learned residual flow + ROI-sum Poisson prior on DnCNN output.
# Residual flow arch is UNCHANGED (sq|unc|unc|gain|unc|unc|gain|unc|unc|usq).
#
# Prior design:
#   - Detect atom candidates as local maxima of the 5x5 sliding sum exceeding
#     `--prior_sum_threshold_photon`.
#   - Apply continuous Poisson NLL with rate λ_atom = expected TOTAL photon
#     count per atom (the <photon> column from make_fidelity_table.py).
#
# Turn off without editing code: drop `--use_prior_flow` or set `--lmbda_prior 0`.
# Existing run_learned_many.sh / run_basden_many.sh are untouched.
#

set -e

TIMES=("5ms" "4ms" "8ms" "10ms" "12ms" "14ms" "16ms" "20ms")

DATA_ROOT="./data_atom"
LOG_ROOT="experiments/paper"

for TIME in "${TIMES[@]}"
do
    # Normalization + expected atom ROI-sum photon rate per exposure.
    # LAMBDA_ATOM = mean_photon from make_fidelity_table.py
    # SUM_THLD = ~0.78 × LAMBDA_ATOM  (tight: only strong atom candidates
    #           pass, minimizing BG false positives that would dilute the
    #           detected NLL).
    case "$TIME" in
        "20ms") CURRENT_VMIN=473; CURRENT_VMAX=805; LAMBDA_ATOM=20.77; SUM_THLD=16.0 ;;
        "16ms") CURRENT_VMIN=473; CURRENT_VMAX=769; LAMBDA_ATOM=16.67; SUM_THLD=13.0 ;;
        "14ms") CURRENT_VMIN=473; CURRENT_VMAX=754; LAMBDA_ATOM=14.36; SUM_THLD=11.0 ;;
        "12ms") CURRENT_VMIN=473; CURRENT_VMAX=731; LAMBDA_ATOM=12.33; SUM_THLD=9.5  ;;
        "10ms") CURRENT_VMIN=473; CURRENT_VMAX=702; LAMBDA_ATOM=10.40; SUM_THLD=8.0  ;;
        "8ms")  CURRENT_VMIN=473; CURRENT_VMAX=678; LAMBDA_ATOM=8.39;  SUM_THLD=6.5  ;;
        "5ms")  CURRENT_VMIN=473; CURRENT_VMAX=647; LAMBDA_ATOM=6.44;  SUM_THLD=5.0  ;;
        "4ms")  CURRENT_VMIN=473; CURRENT_VMAX=654; LAMBDA_ATOM=4.45;  SUM_THLD=3.5  ;;
        *) echo "Error: undefined TIME -> $TIME"; exit 1 ;;
    esac

    ROI_SIZE=5
    LMBDA_PRIOR=1.0   # gentle prior on top of pre-trained DnCNN (warm-start)

    # Warm-start from an already-trained learned checkpoint.
    # This is essential: training prior from scratch causes DnCNN to freeze.
    WARM_CKPT="./experiments/archive/n2nf_learned_only_${TIME}_last.pth"

    echo ""
    echo "=========================================="
    echo "Starting HYBRID-PRIOR training for: $TIME"
    echo "Norm: VMIN=$CURRENT_VMIN, VMAX=$CURRENT_VMAX"
    echo "Arch: sq|unc|unc|gain|unc|unc|gain|unc|unc|usq  (learned residual)"
    echo "Warm-start from: ${WARM_CKPT}"
    echo "ROI-sum Poisson: λ_atom=${LAMBDA_ATOM}ph, sum_thld=${SUM_THLD}ph,"
    echo "                 roi=${ROI_SIZE}x${ROI_SIZE}, weight=${LMBDA_PRIOR}"
    echo "=========================================="

    if [ ! -f "$WARM_CKPT" ]; then
        echo "Error: warm-start checkpoint not found -> $WARM_CKPT"
        echo "Run run_learned_many.sh first to produce the learned baseline."
        exit 1
    fi

    LOG_DIR_NAME="n2nf_hybrid_prior_${TIME}"

    CURRENT_DATA_PATH="${DATA_ROOT}/${TIME}"
    if [ ! -d "$CURRENT_DATA_PATH" ]; then
        echo "Error: Data directory not found -> $CURRENT_DATA_PATH"
        exit 1
    fi

    # Fine-tuning from warm-start: much shorter than full training
    MIN_EPOCHS=50
    MAX_EPOCHS=150
    EARLY_STOP_PATIENCE=3
    EARLY_STOP_MIN_DELTA=0.0005

    echo ">>> [${TIME}] hybrid fine-tuning, epochs=${MIN_EPOCHS}..${MAX_EPOCHS}"
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
        --init_from "$WARM_CKPT" \
        --n_train_threads 0 \
        --vmin "$CURRENT_VMIN" \
        --vmax "$CURRENT_VMAX" \
        --use_prior_flow \
        --prior_lambda_atom "$LAMBDA_ATOM" \
        --prior_sum_threshold_photon "$SUM_THLD" \
        --prior_roi_size "$ROI_SIZE" \
        --lmbda_prior "$LMBDA_PRIOR"

    sleep 60

    echo ">>> Checkpoints and logs saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
done

echo ""
echo "=========================================="
echo "All HYBRID-PRIOR training sessions completed."
echo "=========================================="
