"""Benchmark learned NoiseFlow: H2D, GPU forward, D2H vs image size.

The learned arch (`sq|unc|unc|gain|unc|unc|gain|unc|unc|usq`) is fully
convolutional, so the same checkpoint runs on any even (H, W). This script
sweeps a list of square sizes and times the three pieces independently with
`torch.cuda.Event`, after a per-size warmup so cudnn.benchmark + JIT settle.

Usage (from this directory):
    python benchmark_learned_noiseflow.py \
        --ckpt experiments/archive/n2nf_learned_only_8ms_last.pth
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.noise_flow import NoiseFlow

ARCH = "sq|unc|unc|gain|unc|unc|gain|unc|unc|usq"
DEFAULT_SIZES = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
WARMUP = 10
REPEAT = 50
BATCH = 1
DEFAULT_EXPOSURE_MS = 8
CKPT_TEMPLATE = "experiments/archive/n2nf_learned_only_{ms}ms_last.pth"
SUPPORTED_EXPOSURES = [4, 5, 8, 10, 12, 14, 16, 20]


def build_model(ckpt_path: Path, device: str) -> NoiseFlow:
    model = NoiseFlow(
        x_shape=[1, 64, 64],
        arch=ARCH,
        flow_permutation=1,
        param_inits=None,
        basden_config=None,
        lu_decomp=True,
        device=device,
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    model.eval()
    return model


@torch.no_grad()
def bench_one(model: NoiseFlow, h: int, w: int, device: str, pin_memory: bool) -> dict:
    cpu_x = torch.randn(BATCH, 1, h, w, pin_memory=pin_memory)

    # --- warmup (lets cudnn.benchmark autotune the conv kernels for this size) ---
    for _ in range(WARMUP):
        x = cpu_x.to(device, non_blocking=True)
        z, _ = model(x)
        z.cpu()
    torch.cuda.synchronize()

    # --- H2D ---
    h2d_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(REPEAT)]
    for s, e in h2d_evs:
        s.record()
        x = cpu_x.to(device, non_blocking=True)
        e.record()
    torch.cuda.synchronize()
    h2d_ms = np.array([s.elapsed_time(e) for s, e in h2d_evs])

    # --- GPU forward (input already on device) ---
    x = cpu_x.to(device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fwd_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(REPEAT)]
    z = None
    for s, e in fwd_evs:
        s.record()
        z, _ = model(x)
        e.record()
    torch.cuda.synchronize()
    fwd_ms = np.array([s.elapsed_time(e) for s, e in fwd_evs])
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    # --- D2H (output stays on device, copy out) ---
    d2h_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(REPEAT)]
    for s, e in d2h_evs:
        s.record()
        _ = z.to("cpu", non_blocking=True)
        e.record()
    torch.cuda.synchronize()
    d2h_ms = np.array([s.elapsed_time(e) for s, e in d2h_evs])

    bytes_per_img = BATCH * 1 * h * w * cpu_x.element_size()
    return dict(
        h=h, w=w, n_pix=h * w, mb=bytes_per_img / 1024 ** 2,
        h2d_ms_mean=h2d_ms.mean(), h2d_ms_std=h2d_ms.std(),
        h2d_gbps=bytes_per_img / (h2d_ms.mean() * 1e-3) / 1024 ** 3,
        fwd_ms_mean=fwd_ms.mean(), fwd_ms_std=fwd_ms.std(),
        fwd_us_per_kpix=fwd_ms.mean() * 1e3 / (h * w / 1000),
        d2h_ms_mean=d2h_ms.mean(), d2h_ms_std=d2h_ms.std(),
        d2h_gbps=bytes_per_img / (d2h_ms.mean() * 1e-3) / 1024 ** 3,
        peak_mb=peak_mb,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure-ms", type=int, default=DEFAULT_EXPOSURE_MS,
                    choices=SUPPORTED_EXPOSURES,
                    help=f"Exposure time (ms) — selects checkpoint from "
                         f"`{CKPT_TEMPLATE}`. Default: {DEFAULT_EXPOSURE_MS}.")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Path to learned NoiseFlow checkpoint (.pth). "
                         "If omitted, derived from --exposure-ms.")
    ap.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES,
                    help=f"Square image sizes to sweep (default: {DEFAULT_SIZES})")
    ap.add_argument("--no-pin", action="store_true",
                    help="Disable pin_memory on the CPU tensor")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path. Default: "
                         "`benchmark_learned_noiseflow_{ms}ms.csv`.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.ckpt is None:
        args.ckpt = script_dir / CKPT_TEMPLATE.format(ms=args.exposure_ms)
    if args.out is None:
        args.out = script_dir / f"benchmark_learned_noiseflow_{args.exposure_ms}ms.csv"
    if not args.ckpt.exists():
        sys.exit(f"checkpoint not found: {args.ckpt}")

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available — this benchmark needs a GPU.")
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"ckpt     : {args.ckpt}")
    print(f"arch     : {ARCH}")
    print(f"batch    : {BATCH}")
    print(f"warmup   : {WARMUP}, repeat: {REPEAT}")
    print()

    model = build_model(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"learned NoiseFlow params: {n_params:,}\n")

    print(f"{'size':>5} {'MB':>6} | "
          f"{'H2D ms':>14} {'GB/s':>6} | "
          f"{'fwd ms':>14} {'us/kpix':>8} | "
          f"{'D2H ms':>14} {'GB/s':>6} | "
          f"{'peak MB':>8}")
    print("-" * 110)

    rows = []
    for sz in args.sizes:
        if sz % 2:
            print(f"  {sz}: skip (must be even for sq/usq)")
            continue
        try:
            r = bench_one(model, sz, sz, device, pin_memory=not args.no_pin)
        except RuntimeError as e:
            print(f"  {sz}x{sz}: OOM / RuntimeError — {e}")
            torch.cuda.empty_cache()
            break
        rows.append(r)
        print(f"{sz:>5} {r['mb']:>6.2f} | "
              f"{r['h2d_ms_mean']:>7.3f}±{r['h2d_ms_std']:5.3f} {r['h2d_gbps']:>5.2f} | "
              f"{r['fwd_ms_mean']:>7.3f}±{r['fwd_ms_std']:5.3f} {r['fwd_us_per_kpix']:>7.2f} | "
              f"{r['d2h_ms_mean']:>7.3f}±{r['d2h_ms_std']:5.3f} {r['d2h_gbps']:>5.2f} | "
              f"{r['peak_mb']:>7.1f}")

    if rows:
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved CSV: {args.out.resolve()}")


if __name__ == "__main__":
    main()
