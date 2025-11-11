import argparse
import random
from pathlib import Path

import tifffile


def _build_split_plan(num_items: int, ratios: tuple[int, int, int]) -> list[str]:
    total = sum(ratios)
    counts = [num_items * r // total for r in ratios]
    remainder = num_items - sum(counts)
    for i in range(remainder):
        counts[i % len(counts)] += 1
    plan = []
    for name, count in zip(("train", "val", "test"), counts):
        plan.extend([name] * count)
    return plan


def main():
    parser = argparse.ArgumentParser(description="Split multi-page TIF into Noise2NoiseFlow custom layout.")
    parser.add_argument("tif_path", type=Path, help="Input multi-page TIF file.")
    parser.add_argument("output_root", type=Path, help="Dataset root that will receive train/val/test splits.")
    parser.add_argument("--start_index", type=int, default=1, help="Starting scene index.")
    parser.add_argument("--ratios", type=int, nargs=3, default=(6, 2, 2), metavar=("train", "val", "test"),
                        help="Split ratios for train/val/test.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle page order before splitting.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed when shuffling is enabled.")
    parser.add_argument("--shots-per-scene", type=int, default=2,
                        help="Number of consecutive TIFF pages that make up one scene (default: 2 for a/b pairs).")
    args = parser.parse_args()

    if not args.tif_path.is_file():
        raise FileNotFoundError(args.tif_path)

    with tifffile.TiffFile(args.tif_path) as tif:
        num_pages = len(tif.pages)
        if num_pages == 0:
            raise ValueError("Input TIF contains no frames.")

        shots = args.shots_per_scene
        if shots <= 0:
            raise ValueError("--shots-per-scene must be a positive integer.")
        if shots > 26:
            raise ValueError("--shots-per-scene must be <= 26 to allow alphabetical file naming.")
        if num_pages % shots != 0:
            raise ValueError(f"Input TIF has {num_pages} pages which is not divisible by shots-per-scene={shots}.")

        scene_groups = [list(range(i * shots, (i + 1) * shots)) for i in range(num_pages // shots)]
        if args.shuffle:
            random.seed(args.seed)
            random.shuffle(scene_groups)

        split_plan = _build_split_plan(len(scene_groups), tuple(args.ratios))
        roots = {name: (args.output_root / name) for name in ("train", "val", "test")}
        for root in roots.values():
            root.mkdir(parents=True, exist_ok=True)

        counts = {"train": 0, "val": 0, "test": 0}
        for offset, group in enumerate(scene_groups):
            subset = split_plan[offset]
            scene_id = args.start_index + offset
            scene_dir = roots[subset] / f"scene_{scene_id:04d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            label_names = [chr(ord('a') + i) for i in range(len(group))]
            for label, page_idx in zip(label_names, group):
                frame = tif.pages[page_idx].asarray()
                tifffile.imwrite(scene_dir / f"{label}.tif", frame)
            counts[subset] += 1

    print("Created scenes:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {counts[split_name]}")


if __name__ == "__main__":
    main()