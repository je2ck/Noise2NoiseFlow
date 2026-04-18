"""
Build Noise2NoiseFlow training dataset from real atom images per exposure time.

Source (data-prep/):
  <base_dir>/raw/<ms>_data_rot90_filtered.tif   : 2N frames. even idx = "a" exposure,
                                                 odd idx = "b" exposure (N pairs)
  <base_dir>/test.tif                            : test set (last ~40% of pairs, A frames)
  <base_dir>/test_second.tif                     : test set (B frames, matched to test.tif)

Split (inferred from existing 8ms layout: train.tif / test.tif sequential blocks):
  train  : pairs [0                  : N_train       )
  val    : pairs [N_train            : N_train+N_val )
  test   : pairs [N_train+N_val      : N_total       )   ← matches existing test.tif/test_second.tif

  where  N_test = len(test.tif)
         N_val  = N_test // 2                       (matches 8ms's 40/20/40 = 3191/1595/3190)
         N_train = N_total - N_val - N_test

Target (data_atom/<ms>/):
  train/scene_NNNN/{a,b}.tif   : single-frame 64x64 uint16
  val/scene_NNNN/{a,b}.tif
  test/scene_NNNN/{a,b}.tif

Usage:
  python gen_from_atom_mock.py                 # all 8 exposures
  python gen_from_atom_mock.py --times 5ms 8ms # subset
  python gen_from_atom_mock.py --dry_run       # print plan only
"""

import argparse
import os
import sys

import numpy as np
from tifffile import imread, imwrite


# ------------------------------------------------------------
# Per-exposure data directories (matches infer_tif.sh / run_basden_many.sh)
# ------------------------------------------------------------
BASE_DIRS = {
    '4ms':  '../../data-prep/data/4ms_array_bs',
    '5ms':  '../../data-prep/data/5ms_array_bs',
    '8ms':  '../../data-prep/data/20260108_8_8_60_array_bs',
    '10ms': '../../data-prep/data/20260108_10_10_60_array_bs',
    '12ms': '../../data-prep/data/20260108_12_12_60_array_bs',
    '14ms': '../../data-prep/data/20260105_14_14_60_array_bs',
    '16ms': '../../data-prep/data/20260105_16_16_60_array_bs',
    '20ms': '../../data-prep/data/20260105_20_20_60_array_bs',
}

OUT_ROOT = './data_atom'   # matches run_basden_many.sh DATA_ROOT


# ------------------------------------------------------------
def build_one(ms, base_dir, out_root, dry_run=False, verify=True,
              val_policy='half_test', val_value=None):
    print(f"\n========== {ms} ==========")
    src = os.path.join(base_dir, 'raw', f'{ms}_data_rot90_filtered.tif')
    test_path = os.path.join(base_dir, 'test.tif')

    if not os.path.exists(src):
        print(f"  [skip] source not found: {src}")
        return

    # Load: paired frames interleaved (A0 B0 A1 B1 ...)
    print(f"  loading {src}")
    data = imread(src)   # (2N, H, W), uint16
    if data.ndim != 3:
        raise ValueError(f"Expected (N,H,W), got {data.shape}")
    if data.shape[0] % 2 != 0:
        print(f"  [WARN] odd number of frames: {data.shape[0]} — dropping last")
        data = data[:-1]
    firsts = data[0::2]    # (N_pairs, H, W)
    seconds = data[1::2]   # (N_pairs, H, W)
    N_total = firsts.shape[0]
    H, W = firsts.shape[1:]
    print(f"    total pairs = {N_total}  ({H}x{W})")

    # Infer split from existing test.tif length
    if not os.path.exists(test_path):
        # Fall back to fixed 40/20/40
        N_test = int(N_total * 0.4)
        print(f"  [note] test.tif not found → default split 40/20/40 (N_test = {N_test})")
    else:
        n_test_frames = imread(test_path).shape[0]
        N_test = n_test_frames
        print(f"  test.tif frames = {N_test}")

    # Val sizing policy
    if val_policy == 'half_test':
        N_val = N_test // 2                         # 8~20ms observed pattern
    elif val_policy == 'fraction':
        N_val = int(round((N_total - N_test) * val_value))
    elif val_policy == 'explicit':
        N_val = int(val_value)
    elif val_policy == 'zero':
        N_val = 0
    else:
        raise ValueError(f"Unknown val_policy: {val_policy}")
    N_train = N_total - N_val - N_test
    if N_train <= 0:
        raise ValueError(f"N_train={N_train} <= 0; bad split. N_total={N_total}, N_test={N_test}, N_val={N_val}")
    print(f"    split  train={N_train}  val={N_val}  test={N_test}  "
          f"({N_train/N_total*100:.1f}/{N_val/N_total*100:.1f}/{N_test/N_total*100:.1f})")

    # Verification: test.tif[0] should equal firsts[N_train + N_val]
    if verify and os.path.exists(test_path):
        t = imread(test_path)
        test_start = N_train + N_val
        if test_start < N_total:
            ok = np.array_equal(firsts[test_start].astype(float), t[0].astype(float))
            if ok:
                print(f"    [verify] test.tif[0] == firsts[{test_start}]  ✓")
            else:
                print(f"    [verify] ✗ test.tif[0] != firsts[{test_start}] — split might be off.")
                # try to find actual test_start by matching test[0]
                for cand in range(N_total):
                    if np.array_equal(firsts[cand].astype(float), t[0].astype(float)):
                        print(f"             (hint: test.tif[0] == firsts[{cand}]  → shift split)")
                        break

    if dry_run:
        print("    [dry_run] skipping file write")
        return

    # Write per-split scene dirs
    splits = {
        'train': (0,              N_train),
        'val':   (N_train,        N_train + N_val),
        'test':  (N_train + N_val, N_total),
    }

    target_root = os.path.join(out_root, ms)
    for split_name, (lo, hi) in splits.items():
        split_dir = os.path.join(target_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"    writing {split_name:<5}  [{lo:5d} : {hi:5d})  → {split_dir}")
        n = hi - lo
        for local_i in range(n):
            global_i = lo + local_i
            scene = os.path.join(split_dir, f'scene_{local_i:05d}')
            os.makedirs(scene, exist_ok=True)
            imwrite(os.path.join(scene, 'a.tif'), firsts[global_i])
            imwrite(os.path.join(scene, 'b.tif'), seconds[global_i])
            if (local_i + 1) % 1000 == 0:
                print(f"        {split_name} {local_i+1}/{n}")
    print(f"  ✓ {ms} done")


# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--times', nargs='+', default=None,
                   help='특정 노광만 (예: --times 5ms 8ms). 기본: 전체 8개')
    p.add_argument('--out_root', type=str, default=OUT_ROOT,
                   help=f'저장 루트. 기본: {OUT_ROOT}')
    p.add_argument('--dry_run', action='store_true',
                   help='파일 안 쓰고 split 계획만 출력')
    p.add_argument('--no_verify', action='store_true',
                   help='test.tif 와 일치 여부 검증 끄기 (bit-exact 비교)')
    p.add_argument('--val_policy', choices=['half_test', 'fraction', 'explicit', 'zero'],
                   default='half_test',
                   help='val 크기 결정 방식. '
                        'half_test(기본,=N_test/2) / fraction(non-test 의 비율) / '
                        'explicit(절대 개수) / zero(val 없음)')
    p.add_argument('--val_value', type=float, default=None,
                   help='val_policy 가 fraction 이면 0~1 사이 비율, explicit 이면 개수')
    args = p.parse_args()

    if args.val_policy in ('fraction', 'explicit') and args.val_value is None:
        p.error(f'--val_policy {args.val_policy} 쓸 때는 --val_value 필요')

    times = args.times if args.times else list(BASE_DIRS.keys())
    print(f"Exposure times: {times}")
    print(f"Output root   : {args.out_root}")
    print(f"Dry run       : {args.dry_run}")

    for ms in times:
        if ms not in BASE_DIRS:
            print(f"[skip] {ms} not in BASE_DIRS")
            continue
        build_one(ms, BASE_DIRS[ms], args.out_root,
                  dry_run=args.dry_run, verify=not args.no_verify,
                  val_policy=args.val_policy, val_value=args.val_value)

    print("\n==========================================")
    print("All done. Run training with:")
    print("  bash run_basden_many.sh")
    print("==========================================")


if __name__ == '__main__':
    main()
