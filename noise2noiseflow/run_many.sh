#!/bin/bash

# 에러 발생 시 중단
set -e

# 학습할 시간 리스트
TIMES=("2ms" "3ms" "4ms" "5ms" "6ms" "8ms")

# 데이터가 위치한 루트 폴더 (이전에 생성한 mock data의 서브 폴더 경로)
DATA_ROOT="./data-mock-hybrid" 
LOG_ROOT="experiments/paper" # 기본 로그 저장 경로 (train_atom.py 내부 로직 준수)

for TIME in "${TIMES[@]}"
do
    echo ""
    echo "=========================================="
    echo "Starting training for exposure time: $TIME"
    echo "=========================================="

    # [수정된 부분]: 로그 디렉토리를 time별로 분리
    # 예: n2nf_2ms, n2nf_3ms
    LOG_DIR_NAME="n2nf_${TIME}" 

    # 데이터 경로 확인 (예: ./data_by_time/2ms)
    CURRENT_DATA_PATH="${DATA_ROOT}/${TIME}" 
    
    if [ ! -d "$CURRENT_DATA_PATH" ]; then
        echo "Error: Data directory not found -> $CURRENT_DATA_PATH"
        exit 1
    fi

    # Python 명령어 실행
    python train_atom.py \
        --arch "unc|unc|unc|unc|gain|unc|unc|unc|unc" \
        --sidd_path "$CURRENT_DATA_PATH" \
        --epochs 200 \
        --n_batch_train 8 \
        --n_batch_test 8\
        --n_patches_per_image 1 \
        --patch_height 64 \
        --patch_sampling uniform \
        --n_channels 1 \
        --epochs_full_valid 10 \
        --lu_decomp \
        --logdir "$LOG_DIR_NAME" \
        --lmbda 262144 \
        --no_resume \
        --n_train_threads 0
    
    sleep 60

    echo ">>> Checkpoints and logs saved to ${LOG_ROOT}/${LOG_DIR_NAME}"
done

echo ""
echo "=========================================="
echo "All training sessions completed successfully!"
echo "=========================================="