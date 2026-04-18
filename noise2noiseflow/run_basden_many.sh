#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

# 학습할 시간 리스트
# TIMES=("20ms" "16ms" "14ms" "12ms" "10ms" "8ms")
TIMES=("5ms" "4ms" "8ms" "10ms" "12ms" "14ms" "16ms" "20ms") # 전체 시간 포함 버전

# 데이터가 위치한 루트 폴더
DATA_ROOT="./data_atom" 
LOG_ROOT="experiments/paper"

for TIME in "${TIMES[@]}"
do
    # --------------------------------------------------------------
    # 1. 시간별 파라미터 매핑 (VMIN, VMAX 및 Basden 파라미터)
    # --------------------------------------------------------------
    # Basden params: log-space dark fit (σ, bias) + atom-image λ refit (stray light 흡수)
    # sensitivity=4.15, em_gain=300 고정
    case "$TIME" in
        "20ms")
            CURRENT_VMIN=473
            CURRENT_VMAX=805
            CURRENT_BIAS=500.364
            CURRENT_SIGMA=9.085
            CURRENT_GAIN=300.00
            CURRENT_CIC=0.01583
            ;;
        "16ms")
            CURRENT_VMIN=473
            CURRENT_VMAX=769
            CURRENT_BIAS=500.174
            CURRENT_SIGMA=9.070
            CURRENT_GAIN=300.00
            CURRENT_CIC=0.01451
            ;;
        "14ms")
            CURRENT_VMIN=473
            CURRENT_VMAX=754
            CURRENT_BIAS=500.539
            CURRENT_SIGMA=9.041
            CURRENT_GAIN=300.00
            CURRENT_CIC=0.01408
            ;;
        "12ms")
            CURRENT_VMIN=473
            CURRENT_VMAX=731
            CURRENT_BIAS=500.221
            CURRENT_SIGMA=9.010
            CURRENT_GAIN=300.00
            CURRENT_CIC=0.01340
            ;;
        "10ms")
            CURRENT_VMIN=473
            CURRENT_VMAX=702
            CURRENT_BIAS=500.079
            CURRENT_SIGMA=8.949
            CURRENT_GAIN=300.00
            CURRENT_CIC=0.01192
            ;;
        "8ms")
            CURRENT_VMIN=473
            CURRENT_VMAX=678
            CURRENT_BIAS=500.223
            CURRENT_SIGMA=8.936
            CURRENT_GAIN=300.00
            CURRENT_CIC=0.01119
            ;;
        "5ms")
            CURRENT_VMIN=473
            CURRENT_VMAX=647
            CURRENT_BIAS=500.182
            CURRENT_SIGMA=9.019
            CURRENT_GAIN=300.00
            CURRENT_CIC=0.01003
            ;;
        "4ms")
            CURRENT_VMIN=473
            CURRENT_VMAX=654
            CURRENT_BIAS=500.295
            CURRENT_SIGMA=9.015
            CURRENT_GAIN=300.00
            CURRENT_CIC=0.01094
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
    LOG_DIR_NAME="n2nf_real_bs_basden_only_${TIME}" 

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
        --arch "basden|cond" \
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

    # # 로그 디렉토리 이름 (예: n2nf_basden_2ms)
    # LOG_DIR_NAME="n2nf_real_basden_hybrid_${TIME}" 

    # if [ ! -d "$CURRENT_DATA_PATH" ]; then
    #     echo "Error: Data directory not found -> $CURRENT_DATA_PATH"
    #     exit 1
    # fi

    # # --------------------------------------------------------------
    # # 2. Python 실행 (Basden 파라미터 인자 추가)
    # # --------------------------------------------------------------
    # python train_atom.py \
    #     --arch "basden|unc|unc|unc|unc|gain|unc|unc|unc|unc" \
    #     --sidd_path "$CURRENT_DATA_PATH" \
    #     --epochs 200 \
    #     --n_batch_train 8 \
    #     --n_batch_test 8 \
    #     --n_patches_per_image 1 \
    #     --patch_height 64 \
    #     --patch_sampling uniform \
    #     --n_channels 1 \
    #     --epochs_full_valid 10 \
    #     --lu_decomp \
    #     --logdir "$LOG_DIR_NAME" \
    #     --lmbda 262144 \
    #     --no_resume \
    #     --n_train_threads 0 \
    #     --vmin "$CURRENT_VMIN" \
    #     --vmax "$CURRENT_VMAX" \
    #     --basden_bias_offset "$CURRENT_BIAS" \
    #     --basden_readout_sigma "$CURRENT_SIGMA" \
    #     --basden_em_gain "$CURRENT_GAIN" \
    #     --basden_cic_lambda "$CURRENT_CIC"
    
    # sleep 60

    # echo ">>> Checkpoints and logs saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
done

echo ""
echo "=========================================="
echo "All training sessions completed successfully!"
echo "=========================================="