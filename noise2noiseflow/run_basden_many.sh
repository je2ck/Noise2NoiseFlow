#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

# 학습할 시간 리스트
TIMES=("4ms" "5ms" "6ms" "2ms" "3ms" "8ms")

# 데이터가 위치한 루트 폴더
DATA_ROOT="./data-mock-hybrid" 
LOG_ROOT="experiments/paper"

for TIME in "${TIMES[@]}"
do
    # --------------------------------------------------------------
    # 1. 시간별 VMIN, VMAX 매핑 (이미지 테이블 참조)
    # --------------------------------------------------------------
    case "$TIME" in
        "2ms")
            CURRENT_VMIN=409
            CURRENT_VMAX=532
            ;;
        "3ms")
            CURRENT_VMIN=409
            CURRENT_VMAX=539
            ;;
        "4ms")
            CURRENT_VMIN=409
            CURRENT_VMAX=548
            ;;
        "5ms")
            CURRENT_VMIN=409
            CURRENT_VMAX=557
            ;;
        "6ms")
            CURRENT_VMIN=409
            CURRENT_VMAX=568
            ;;
        "8ms")
            CURRENT_VMIN=409
            CURRENT_VMAX=590
            ;;
        *)
            echo "Error: 정의되지 않은 시간 포맷입니다 -> $TIME"
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "Starting training for: $TIME"
    echo "Parameters: VMIN=$CURRENT_VMIN, VMAX=$CURRENT_VMAX"
    echo "=========================================="

    # 로그 디렉토리 이름 (예: n2nf_2ms)
    LOG_DIR_NAME="n2nf_basden_${TIME}" 

    # 실제 데이터 경로 (예: ./data-mock-hybrid/2ms)
    CURRENT_DATA_PATH="${DATA_ROOT}/${TIME}" 
    
    if [ ! -d "$CURRENT_DATA_PATH" ]; then
        echo "Error: Data directory not found -> $CURRENT_DATA_PATH"
        exit 1
    fi

    # --------------------------------------------------------------
    # 2. Python 실행 (--vmin, --vmax 인자 추가됨)
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
        --vmax "$CURRENT_VMAX"
    
    sleep 60

    echo ">>> Checkpoints and logs saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
done

echo ""
echo "=========================================="
echo "All training sessions completed successfully!"
echo "=========================================="