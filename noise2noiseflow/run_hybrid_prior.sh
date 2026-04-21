#!/bin/bash
#
# Learned residual flow + continuous-Poisson prior on DnCNN output.
# Residual flow arch is UNCHANGED (sq|unc|unc|gain|unc|unc|gain|unc|unc|usq);
# adds a second NLL term on x_hat for atom-like pixels.
#
# λ_atom per exposure = mean atom photon count (from fidelity table).
# λ_prior = 0.1 gives the Poisson term ~0.1× weight of residual NLL.
#
# Turn this run OFF without touching the code: drop `--use_prior_flow` or set
# `--lmbda_prior 0`. Existing run_learned_many.sh / run_basden_many.sh are
# untouched.
#

set -e

TIMES=("5ms" "4ms" "8ms" "10ms" "12ms" "14ms" "16ms" "20ms")

DATA_ROOT="./data_atom"
LOG_ROOT="experiments/paper"

for TIME in "${TIMES[@]}"
do
    # Normalization + expected atom photon rate per exposure.
    # λ_atom ≈ mean_photon from make_fidelity_table.py
    case "$TIME" in
        "20ms") CURRENT_VMIN=473; CURRENT_VMAX=805; LAMBDA_ATOM=20.77 ;;
        "16ms") CURRENT_VMIN=473; CURRENT_VMAX=769; LAMBDA_ATOM=16.67 ;;
        "14ms") CURRENT_VMIN=473; CURRENT_VMAX=754; LAMBDA_ATOM=14.36 ;;
        "12ms") CURRENT_VMIN=473; CURRENT_VMAX=731; LAMBDA_ATOM=12.33 ;;
        "10ms") CURRENT_VMIN=473; CURRENT_VMAX=702; LAMBDA_ATOM=10.40 ;;
        "8ms")  CURRENT_VMIN=473; CURRENT_VMAX=678; LAMBDA_ATOM=8.39 ;;
        "5ms")  CURRENT_VMIN=473; CURRENT_VMAX=647; LAMBDA_ATOM=6.44 ;;
        "4ms")  CURRENT_VMIN=473; CURRENT_VMAX=654; LAMBDA_ATOM=4.45 ;;
        *) echo "Error: undefined TIME -> $TIME"; exit 1 ;;
    esac

    # Atom detection threshold in photon units. 1.5 photons works for all
    # exposures: well above 1σ photon shot noise of background, below the
    # smallest atom mean (4.45 at 4ms).
    ATOM_THLD_PHOTON=1.5
    LMBDA_PRIOR=0.1

    echo ""
    echo "=========================================="
    echo "Starting HYBRID-PRIOR training for: $TIME"
    echo "Norm: VMIN=$CURRENT_VMIN, VMAX=$CURRENT_VMAX"
    echo "Arch: sq|unc|unc|gain|unc|unc|gain|unc|unc|usq  (learned residual)"
    echo "Prior: Poisson(λ=${LAMBDA_ATOM}), thld=${ATOM_THLD_PHOTON} ph, weight=${LMBDA_PRIOR}"
    echo "=========================================="

    LOG_DIR_NAME="n2nf_hybrid_prior_${TIME}"

    CURRENT_DATA_PATH="${DATA_ROOT}/${TIME}"
    if [ ! -d "$CURRENT_DATA_PATH" ]; then
        echo "Error: Data directory not found -> $CURRENT_DATA_PATH"
        exit 1
    fi

    MIN_EPOCHS=200
    MAX_EPOCHS=500
    EARLY_STOP_PATIENCE=5
    EARLY_STOP_MIN_DELTA=0.0005

    echo ">>> [${TIME}] hybrid (learned + Poisson prior), epochs=${MIN_EPOCHS}..${MAX_EPOCHS}"
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
        --vmax "$CURRENT_VMAX" \
        --use_prior_flow \
        --prior_lambda_atom "$LAMBDA_ATOM" \
        --prior_atom_threshold_photon "$ATOM_THLD_PHOTON" \
        --lmbda_prior "$LMBDA_PRIOR"

    sleep 60

    echo ">>> Checkpoints and logs saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
done

echo ""
echo "=========================================="
echo "All HYBRID-PRIOR training sessions completed."
echo "=========================================="
