#!/bin/bash
#
# 모든 노광시간에 대해 atom TIF + positions 를 써서 λ 재피팅.
# σ, bias 는 dark-frame fit 결과(fit_results/)에서 읽어와 고정.
# 결과: fit_results_atom/<TIME>/emccd_fit_atom_<TIME>.{png,json}
# 요약: fit_results_atom/summary.csv
#
# 사용:  bash run_fit_basden_atom_all.sh
#        MASK_RADIUS=4 bash run_fit_basden_atom_all.sh   # 마스크 반경 변경
#        FIT_BIAS=1 bash run_fit_basden_atom_all.sh      # bias 도 fit

set -e

DARK_RESULTS_DIR="fit_results"             # 기존 dark-frame fit (σ 고정용)
OUT_ROOT="fit_results_atom"
mkdir -p "$OUT_ROOT"

MASK_RADIUS="${MASK_RADIUS:-3}"
FIT_BIAS_FLAG=""
if [ "${FIT_BIAS:-0}" = "1" ]; then
    FIT_BIAS_FLAG="--fit_bias"
fi

# 고정 물리상수
SENSITIVITY=4.15
EM_GAIN=300.0

TIMES=("4ms" "5ms" "8ms" "10ms" "12ms" "14ms" "16ms" "20ms")

# atom TIF 경로 매핑 (infer_tif.sh 의 DATA_DIR 기준)
tif_path_for() {
    case "$1" in
        4ms)  echo "../../data-prep/data/4ms_array_bs/test_fp_filtered.tif" ;;
        5ms)  echo "../../data-prep/data/5ms_array_bs/test_fp_filtered.tif" ;;
        8ms)  echo "../../data-prep/data/20260108_8_8_60_array_bs/test_fp_filtered.tif" ;;
        10ms) echo "../../data-prep/data/20260108_10_10_60_array_bs/test_fp_filtered.tif" ;;
        12ms) echo "../../data-prep/data/20260108_12_12_60_array_bs/test_fp_filtered.tif" ;;
        14ms) echo "../../data-prep/data/20260105_14_14_60_array_bs/test_fp_filtered.tif" ;;
        16ms) echo "../../data-prep/data/20260105_16_16_60_array_bs/test_fp_filtered.tif" ;;
        20ms) echo "../../data-prep/data/20260105_20_20_60_array_bs/test_fp_filtered.tif" ;;
        *) echo "" ;;
    esac
}

pos_path_for() {
    case "$1" in
        4ms)  echo "../../data-prep/data/4ms_array_bs/positions_data.npy" ;;
        5ms)  echo "../../data-prep/data/5ms_array_bs/positions_data.npy" ;;
        8ms)  echo "../../data-prep/data/20260108_8_8_60_array_bs/positions_data.npy" ;;
        10ms) echo "../../data-prep/data/20260108_10_10_60_array_bs/positions_data.npy" ;;
        12ms) echo "../../data-prep/data/20260108_12_12_60_array_bs/positions_data.npy" ;;
        14ms) echo "../../data-prep/data/20260105_14_14_60_array_bs/positions_data.npy" ;;
        16ms) echo "../../data-prep/data/20260105_16_16_60_array_bs/positions_data.npy" ;;
        20ms) echo "../../data-prep/data/20260105_20_20_60_array_bs/positions_data.npy" ;;
        *) echo "" ;;
    esac
}

for TIME in "${TIMES[@]}"; do
    TIF="$(tif_path_for "$TIME")"
    POS="$(pos_path_for "$TIME")"
    DARK_JSON="${DARK_RESULTS_DIR}/${TIME}/emccd_fit_${TIME}.json"
    OUT_DIR="${OUT_ROOT}/${TIME}"

    echo ""
    echo "=========================================="
    echo "[${TIME}]  atom TIF  = ${TIF}"
    echo "          positions = ${POS}"
    echo "          dark fit  = ${DARK_JSON}"
    echo "=========================================="

    if [ ! -f "$TIF" ]; then
        echo "  [skip] atom TIF not found"
        continue
    fi
    if [ ! -f "$POS" ]; then
        echo "  [skip] positions not found"
        continue
    fi
    if [ ! -f "$DARK_JSON" ]; then
        echo "  [skip] prior dark-frame fit JSON not found. Run run_fit_basden_all.sh first."
        continue
    fi

    # σ, bias 를 dark-frame fit JSON 에서 추출
    SIGMA=$(python -c "import json; print(json.load(open('${DARK_JSON}'))['readout_sigma'])")
    BIAS=$(python -c "import json; print(json.load(open('${DARK_JSON}'))['bias_offset'])")

    echo "  σ fixed = ${SIGMA}, bias = ${BIAS}"

    python fit_basden_from_atom_images.py \
        --tif "$TIF" \
        --positions "$POS" \
        --sigma "$SIGMA" \
        --bias "$BIAS" \
        --em_gain "$EM_GAIN" \
        --sensitivity "$SENSITIVITY" \
        --mask_radius "$MASK_RADIUS" \
        --out_dir "$OUT_DIR" \
        --tag "$TIME" \
        $FIT_BIAS_FLAG
done

# --------------------------------------------
# 요약 CSV (dark-frame fit 과 비교)
# --------------------------------------------
SUMMARY="${OUT_ROOT}/summary.csv"
python - <<PY
import glob, json, os, csv

rows = []
for f in sorted(glob.glob("${OUT_ROOT}/*/emccd_fit_atom_*.json")):
    with open(f) as fp:
        d = json.load(fp)
    # 같은 시간의 dark fit 찾아 비교
    tag = d.get('tag')
    dark_path = f"${DARK_RESULTS_DIR}/{tag}/emccd_fit_{tag}.json"
    if os.path.exists(dark_path):
        with open(dark_path) as fp:
            dd = json.load(fp)
        d['cic_lambda_dark'] = dd.get('cic_lambda')
        d['bias_dark'] = dd.get('bias_offset')
    else:
        d['cic_lambda_dark'] = None
        d['bias_dark'] = None
    rows.append(d)

if not rows:
    print("No results found.")
else:
    keys = ['tag', 'bias_offset', 'readout_sigma', 'cic_lambda',
            'cic_lambda_dark', 'bias_dark', 'em_gain', 'sensitivity',
            'n_bg_samples', 'mask_radius', 'fit_bias']
    with open("${SUMMARY}", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nSummary written to ${SUMMARY}")
    print(f"\n{'tag':<8} {'bias':>8} {'sigma':>8} {'λ_atom':>10} {'λ_dark':>10} {'ratio':>7}")
    for r in rows:
        ld = r.get('cic_lambda_dark') or 0
        la = r['cic_lambda']
        ratio = la/ld if ld > 0 else float('nan')
        print(f"{r['tag']:<8} {r['bias_offset']:>8.3f} {r['readout_sigma']:>8.3f} "
              f"{la:>10.5f} {ld:>10.5f} {ratio:>7.2f}x")
PY

echo ""
echo "=========================================="
echo "All atom-image λ refits done → ${OUT_ROOT}/"
echo "=========================================="
