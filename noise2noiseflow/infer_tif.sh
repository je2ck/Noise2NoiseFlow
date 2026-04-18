#!/usr/bin/env bash
set -e

declare -A BASEDIR
BASEDIR[20]="../../data-prep/data/20260105_20_20_60_array_bs"
BASEDIR[16]="../../data-prep/data/20260105_16_16_60_array_bs"
BASEDIR[14]="../../data-prep/data/20260105_14_14_60_array_bs"
BASEDIR[12]="../../data-prep/data/20260108_12_12_60_array_bs"
BASEDIR[10]="../../data-prep/data/20260108_10_10_60_array_bs"
BASEDIR[8]="../../data-prep/data/20260108_8_8_60_array_bs"
BASEDIR[5]="../../data-prep/data/5ms_array_bs"
BASEDIR[4]="../../data-prep/data/4ms_array_bs"
SCRIPT=python\ infer_tif.py
CKPT_DIR=./experiments/archive

# ==========================
# exposure 리스트
# ==========================
# MS_LIST=(20 16 14 12 10 8)
MS_LIST=(5)

for ms in "${MS_LIST[@]}"; do
  DATA_DIR="${BASEDIR[$ms]}"
  echo "======================================"
  echo "Running denoise for ${ms} ms"
  echo "======================================"


  # -------- ms별 파라미터: log-space dark fit (σ, bias) + atom-image λ refit --------
  if [ "$ms" -eq 20 ]; then
    CKPT=${CKPT_DIR}/real_bs_basden_only_20ms_last.pth
    VMIN=473
    VMAX=805
    BIAS=500.364
    READOUT=9.085
    CIC=0.01583

  elif [ "$ms" -eq 8 ]; then
    CKPT=${CKPT_DIR}/real_bs_basden_only_8ms_last.pth
    VMIN=473
    VMAX=678
    BIAS=500.223
    READOUT=8.936
    CIC=0.01119

  elif [ "$ms" -eq 10 ]; then
    CKPT=${CKPT_DIR}/real_bs_basden_only_10ms_last.pth
    VMIN=473
    VMAX=702
    BIAS=500.079
    READOUT=8.949
    CIC=0.01192

  elif [ "$ms" -eq 12 ]; then
    CKPT=${CKPT_DIR}/real_bs_basden_only_12ms_last.pth
    VMIN=473
    VMAX=731
    BIAS=500.221
    READOUT=9.010
    CIC=0.01340

  elif [ "$ms" -eq 14 ]; then
    CKPT=${CKPT_DIR}/real_bs_basden_only_14ms_last.pth
    VMIN=473
    VMAX=754
    BIAS=500.539
    READOUT=9.041
    CIC=0.01408

  elif [ "$ms" -eq 16 ]; then
    CKPT=${CKPT_DIR}/real_bs_basden_only_16ms_last.pth
    VMIN=473
    VMAX=769
    BIAS=500.174
    READOUT=9.070
    CIC=0.01451
  fi
  if [ "$ms" -eq 5 ]; then
    CKPT=${CKPT_DIR}/real_bs_basden_only_5ms_last.pth
    VMIN=473
    VMAX=647
    BIAS=500.182
    READOUT=9.019
    CIC=0.01003

  elif [ "$ms" -eq 4 ]; then
    CKPT=${CKPT_DIR}/real_bs_basden_only_4ms_last.pth
    VMIN=473
    VMAX=654
    BIAS=500.295
    READOUT=9.015
    CIC=0.01094
  fi
  # --------------------------------

  ${SCRIPT} \
    --input ${DATA_DIR}/test_fp_filtered.tif \
    --output ${DATA_DIR}/test_denoised.tif \
    --bg_input ${DATA_DIR}/raw/${ms}ms_data_background.tif \
    --bg_output ${DATA_DIR}/bg_denoised.tif \
    --ckpt ${CKPT} \
    --vmin ${VMIN} \
    --vmax ${VMAX} \
    --basden_bias_offset ${BIAS} \
    --basden_readout_sigma ${READOUT} \
    --basden_cic_lambda ${CIC}

done

echo "✅ All denoising jobs finished."