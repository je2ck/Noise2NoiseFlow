#!/usr/bin/env python
import argparse
import random
from pathlib import Path

import tifffile


def _build_split_plan(num_items: int, ratios: tuple[int, int, int]) -> list[str]:
    """num_items 개의 scene을 주어진 비율로 train/val/test에 배분하는 계획을 만든다."""
    total = sum(ratios)
    counts = [num_items * r // total for r in ratios]
    remainder = num_items - sum(counts)
    for i in range(remainder):
        counts[i % len(counts)] += 1

    plan: list[str] = []
    for name, count in zip(("train", "val", "test"), counts):
        plan.extend([name] * count)
    return plan


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split multi-page TIF (8ms,10ms,20ms repeating) into "
            "Noise2NoiseFlow layout using (8ms,10ms) pairs."
        )
    )
    parser.add_argument("tif_path", type=Path, help="Input multi-page TIF file.")
    parser.add_argument(
        "output_root", type=Path,
        help="Dataset root that will receive train/val/test splits."
    )
    parser.add_argument(
        "--start_index", type=int, default=1,
        help="Starting scene index (scene_XXXX)."
    )
    parser.add_argument(
        "--ratios", type=int, nargs=3, default=(6, 2, 2),
        metavar=("train", "val", "test"),
        help="Split ratios for train/val/test."
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle scene order (triples) before splitting."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed when shuffling is enabled."
    )
    parser.add_argument(
        "--keep-long", action="store_true",
        help="If set, also save the 20ms frame as 'c.tif' in each scene."
    )
    args = parser.parse_args()

    if not args.tif_path.is_file():
        raise FileNotFoundError(args.tif_path)

    # 열어서 페이지 수 확인
    with tifffile.TiffFile(args.tif_path) as tif:
        num_pages = len(tif.pages)
        if num_pages == 0:
            raise ValueError("Input TIF contains no frames.")

        # 8,10,20 이 3개 단위로 반복된다고 가정
        group_size = 3
        if num_pages % group_size != 0:
            raise ValueError(
                f"Input TIF has {num_pages} pages which is not divisible by 3.\n"
                "This script assumes frames come in 8ms,10ms,20ms repeating triples."
            )

        num_scenes = num_pages // group_size

        # 각 scene은 (8ms_idx, 10ms_idx, 20ms_idx) 인덱스 튜플
        scene_triples = [
            (3 * i, 3 * i + 1, 3 * i + 2) for i in range(num_scenes)
        ]

        if args.shuffle:
            random.seed(args.seed)
            random.shuffle(scene_triples)

        # train/val/test 배분 계획
        split_plan = _build_split_plan(num_scenes, tuple(args.ratios))

        # 출력 루트 준비
        roots = {
            name: (args.output_root / name)
            for name in ("train", "val", "test")
        }
        for root in roots.values():
            root.mkdir(parents=True, exist_ok=True)

        counts = {"train": 0, "val": 0, "test": 0}

        for offset, triple in enumerate(scene_triples):
            subset = split_plan[offset]  # "train" / "val" / "test"
            scene_id = args.start_index + offset
            scene_dir = roots[subset] / f"scene_{scene_id:04d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            idx_8, idx_10, idx_20 = triple

            # a.tif ← 8ms frame
            frame_8 = tif.pages[idx_8].asarray()
            tifffile.imwrite(scene_dir / "a.tif", frame_8)

            # b.tif ← 10ms frame
            frame_10 = tif.pages[idx_10].asarray()
            tifffile.imwrite(scene_dir / "b.tif", frame_10)

            # 옵션: 20ms도 같이 저장하고 싶으면 c.tif로 저장
            if args.keep_long:
                frame_20 = tif.pages[idx_20].asarray()
                tifffile.imwrite(scene_dir / "c.tif", frame_20)

            counts[subset] += 1

    print("Created scenes:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {counts[split_name]}")


if __name__ == "__main__":
    main()