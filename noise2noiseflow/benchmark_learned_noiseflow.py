"""Benchmark inference path of the learned N2NF: DnCNN denoiser only.

At inference time only the DnCNN denoiser is used (the NoiseFlow head is a
training-time likelihood module and is bypassed by `model.denoise(...)` →
`self.denoiser(noisy)` in noise2noise_flow.py). This script therefore loads
*only* the DnCNN weights from the checkpoint and times its H2D / GPU
forward / D2H pieces independently with `torch.cuda.Event`, after a
per-size warmup so cudnn.benchmark + JIT settle.

Usage (from this directory):
    python benchmark_learned_noiseflow.py --exposure-ms 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
# `model/` is a namespace package split across `noise2noiseflow/model/`
# and `Noise2NoiseFlow/model/` (where dncnn.py lives).
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
from model.dncnn import DnCNN

DEFAULT_SIZES = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
WARMUP = 10
REPEAT = 50
BATCH = 1
DEFAULT_EXPOSURE_MS = 8
CKPT_TEMPLATE = "experiments/archive/n2nf_learned_only_{ms}ms_last.pth"
SUPPORTED_EXPOSURES = [4, 5, 8, 10, 12, 14, 16, 20]
DNCNN_CHANNELS = 2  # ckpt was trained with C=2 (raw + aux channel; see infer_tif.py)
DNCNN_LAYERS = 25


def build_dncnn(ckpt_path: Path, device: str) -> DnCNN:
    """Load DnCNN weights from a Noise2NoiseFlow checkpoint, ignoring flow keys."""
    model = DnCNN(channels=DNCNN_CHANNELS, num_of_layers=DNCNN_LAYERS).to(device)

    raw = torch.load(ckpt_path, map_location=device)
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd

    # Keep only `denoiser.*` entries and strip the prefix
    prefix = "denoiser."
    dncnn_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not dncnn_sd:
        sys.exit(
            f"no `denoiser.*` keys found in {ckpt_path}; "
            f"keys[:5]={list(sd.keys())[:5]}"
        )
    missing, unexpected = model.load_state_dict(dncnn_sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    model.eval()
    return model


@torch.no_grad()
def bench_one(model: DnCNN, h: int, w: int, device: str, pin_memory: bool) -> dict:
    cpu_x = torch.randn(BATCH, DNCNN_CHANNELS, h, w, pin_memory=pin_memory)

    # --- warmup (cudnn.benchmark autotune for this size) ---
    for _ in range(WARMUP):
        x = cpu_x.to(device, non_blocking=True)
        y = model(x)
        y.cpu()
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

    # --- DnCNN forward (input already on device) ---
    x = cpu_x.to(device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fwd_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(REPEAT)]
    y = None
    for s, e in fwd_evs:
        s.record()
        y = model(x)
        e.record()
    torch.cuda.synchronize()
    fwd_ms = np.array([s.elapsed_time(e) for s, e in fwd_evs])
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    # --- D2H ---
    d2h_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(REPEAT)]
    for s, e in d2h_evs:
        s.record()
        _ = y.to("cpu", non_blocking=True)
        e.record()
    torch.cuda.synchronize()
    d2h_ms = np.array([s.elapsed_time(e) for s, e in d2h_evs])

    bytes_per_img = BATCH * DNCNN_CHANNELS * h * w * cpu_x.element_size()
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
                    help="Path to N2NF checkpoint (.pth). "
                         "If omitted, derived from --exposure-ms.")
    ap.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES,
                    help=f"Square image sizes to sweep (default: {DEFAULT_SIZES})")
    ap.add_argument("--no-pin", action="store_true",
                    help="Disable pin_memory on the CPU tensor")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path. Default: "
                         "`benchmark_dncnn_inference_{ms}ms.csv`.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.ckpt is None:
        args.ckpt = script_dir / CKPT_TEMPLATE.format(ms=args.exposure_ms)
    if args.out is None:
        args.out = script_dir / f"benchmark_dncnn_inference_{args.exposure_ms}ms.csv"
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
    print(f"target   : DnCNN (inference path; NoiseFlow head not run)")
    print(f"channels : {DNCNN_CHANNELS}, layers: {DNCNN_LAYERS}")
    print(f"batch    : {BATCH}")
    print(f"warmup   : {WARMUP}, repeat: {REPEAT}")
    print()

    model = build_dncnn(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DnCNN params: {n_params:,}\n")

    print(f"{'size':>5} {'MB':>6} | "
          f"{'H2D ms':>14} {'GB/s':>6} | "
          f"{'DnCNN ms':>14} {'us/kpix':>8} | "
          f"{'D2H ms':>14} {'GB/s':>6} | "
          f"{'peak MB':>8}")
    print("-" * 110)

    rows = []
    for sz in args.sizes:
        if sz % 2:
            print(f"  {sz}: skip (DnCNN OK on odd, but kept even for parity)")
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
