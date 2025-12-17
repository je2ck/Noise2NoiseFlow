#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

# 학습할 시간 리스트
TIMES=("4ms" "5ms")

# 데이터가 위치한 루트 폴더
DATA_ROOT="./data_atom" 
LOG_ROOT="experiments/paper"

for TIME in "${TIMES[@]}"
do
    # --------------------------------------------------------------
    # 1. 시간별 파라미터 매핑 (VMIN, VMAX 및 Basden 파라미터)
    # --------------------------------------------------------------
    # TODO: 아래의 Basden 파라미터 값들을 실제 피팅 결과값으로 수정하세요.
    case "$TIME" in
        "4ms")
            CURRENT_VMIN=384
            CURRENT_VMAX=677
            # Basden Parameters for 4ms
            CURRENT_BIAS=441.13
            CURRENT_SIGMA=19.64
            CURRENT_GAIN=200.00
            CURRENT_CIC=0.0580
            ;;
        "5ms")
            CURRENT_VMIN=384
            CURRENT_VMAX=634
            # Basden Parameters for 5ms
            CURRENT_BIAS=440.81
            CURRENT_SIGMA=19.69
            CURRENT_GAIN=200.00
            CURRENT_CIC=0.0574
            ;;
        *)
            echo "Error: 정의되지 않은 시간 포맷입니다 -> $TIME"
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "Starting training for: $TIME"
    echo "Norm Params: VMIN=$CURRENT_VMIN, VMAX=$CURRENT_VMAX"
    echo "Basden Params: Bias=$CURRENT_BIAS, Sigma=$CURRENT_SIGMA, Gain=$CURRENT_GAIN, CIC=$CURRENT_CIC"
    echo "=========================================="

    # 로그 디렉토리 이름 (예: n2nf_basden_2ms)
    LOG_DIR_NAME="n2nf_real_basden_only_${TIME}" 

    # 실제 데이터 경로 (예: ./data-mock-hybrid/2ms)
    CURRENT_DATA_PATH="${DATA_ROOT}/${TIME}" 
    
    if [ ! -d "$CURRENT_DATA_PATH" ]; then
        echo "Error: Data directory not found -> $CURRENT_DATA_PATH"
        exit 1
    fi

    # --------------------------------------------------------------
    # 2. Python 실행 (Basden 파라미터 인자 추가)
    # --------------------------------------------------------------
    python train_atom.py \
        --arch "basden" \
        --sidd_path "$CURRENT_DATA_PATH" \
        --epochs 200 \
        --n_batch_train 8 \
        --n_batch_test 8 \
        --n_patches_per_image 1 \
        --patch_height 64 \
        --patch_sampling uniform \
        --n_channels 1 \
        --epochs_full_valid 10 \
        --lu_decomp \
        --logdir "$LOG_DIR_NAME" \
        --lmbda 262144 \
        --no_resume \
        --n_train_threads 0 \
        --vmin "$CURRENT_VMIN" \
        --vmax "$CURRENT_VMAX" \
        --basden_bias_offset "$CURRENT_BIAS" \
        --basden_readout_sigma "$CURRENT_SIGMA" \
        --basden_em_gain "$CURRENT_GAIN" \
        --basden_cic_lambda "$CURRENT_CIC"
    
    sleep 60

    echo ">>> Checkpoints and logs saved to ${LOG_ROOT}/${LOG_DIR_NAME}"

    # 로그 디렉토리 이름 (예: n2nf_basden_2ms)
    LOG_DIR_NAME="n2nf_real_basden_hybrid_${TIME}" 

    if [ ! -d "$CURRENT_DATA_PATH" ]; then
        echo "Error: Data directory not found -> $CURRENT_DATA_PATH"
        exit 1
    fi

    # --------------------------------------------------------------
    # 2. Python 실행 (Basden 파라미터 인자 추가)
    # --------------------------------------------------------------
    python train_atom.py \
        --arch "basden|unc|unc|unc|unc|gain|unc|unc|unc|unc" \
        --sidd_path "$CURRENT_DATA_PATH" \
        --epochs 200 \
        --n_batch_train 8 \
        --n_batch_test 8 \
        --n_patches_per_image 1 \
        --patch_height 64 \
        --patch_sampling uniform \
        --n_channels 1 \
        --epochs_full_valid 10 \
        --lu_decomp \
        --logdir "$LOG_DIR_NAME" \
        --lmbda 262144 \
        --no_resume \
        --n_train_threads 0 \
        --vmin "$CURRENT_VMIN" \
        --vmax "$CURRENT_VMAX" \
        --basden_bias_offset "$CURRENT_BIAS" \
        --basden_readout_sigma "$CURRENT_SIGMA" \
        --basden_em_gain "$CURRENT_GAIN" \
        --basden_cic_lambda "$CURRENT_CIC"
    
    sleep 60

    echo ">>> Checkpoints and logs saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
done

echo ""
echo "=========================================="
echo "All training sessions completed successfully!"
echo "=========================================="