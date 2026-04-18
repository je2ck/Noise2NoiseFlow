"""
Fit Basden CIC lambda from *atom images* by masking out atom positions.

Motivation
----------
이전 `fit_basden_params.py`는 SDB OFF 상태의 pure dark frame 에서 fit 했기 때문에
λ(CIC) 값이 실제 실험 조건(SDB ON)의 stray light 기여를 놓칩니다. 학습·추론 데이터는
SDB ON 조건이므로 이 mismatch가 분류 fidelity 하락의 원인이 됩니다.

해결: 실측 atom TIF 에서 원자 위치 주변을 마스크로 제외하고, 남은 픽셀
(= stray light 포함 background)로 λ를 재피팅. σ, bias, em_gain, sensitivity 는
이전 dark-frame fit 에서 얻은 값으로 **고정**. 논문(Meschede) 방식과 일치:
"p_0 = CIC + stray light"를 한 파라미터로 흡수.

Usage
-----
python fit_basden_from_atom_images.py \
    --tif ../../data-prep/data/5ms_array_bs/test_fp_filtered.tif \
    --positions ../../data-prep/data/5ms_array_bs/positions_data.npy \
    --sigma 8.503 --bias 499.17 \
    --out_dir fit_results_atom --tag 5ms \
    --mask_radius 3

Outputs (out_dir/)
------------------
  emccd_fit_atom_<tag>.png   : histogram + fit + gauss component
  emccd_fit_atom_<tag>.json  : {bias, readout_sigma, cic_lambda, em_gain, sensitivity}
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import i1
from tifffile import imread


# ------------------------------------------------------------
# Basden PDF (fit_basden_params.py 와 동일 형식)
# ------------------------------------------------------------
def basden_pdf(x_adu, bias, sigma_adu, lam, scale, em_gain, sens):
    """
    p(x) = e^{-lam} · N(x; bias, sigma)   (0-photon: Gaussian readout)
         +          Basden(x; g, lam, s)  (>=1 photon: EM-amplified)
    """
    gauss = np.exp(-lam) / (np.sqrt(2 * np.pi) * sigma_adu) \
            * np.exp(-0.5 * ((x_adu - bias) / sigma_adu) ** 2)

    x_e = (x_adu - bias) * sens
    basden = np.zeros_like(x_adu, dtype=np.float64)
    mask = x_e > 1e-2
    if np.any(mask):
        z = x_e[mask]
        term_coef = np.sqrt(lam) / (np.sqrt(em_gain * z) + 1e-12)
        term_exp = np.exp(-(z / em_gain + lam))
        arg_bessel = 2 * np.sqrt(lam * z / em_gain)
        val = term_coef * term_exp * i1(arg_bessel)
        basden[mask] = val * sens

    return scale * (gauss + basden)


# ------------------------------------------------------------
# Mask builders
# ------------------------------------------------------------
def _mask_positions(shape, positions, radius):
    """주어진 (y,x) 점들 주변 (2r+1)x(2r+1) 박스를 False 로 만든 bool 마스크."""
    H, W = shape
    mask = np.ones((H, W), dtype=bool)
    for y, x in positions:
        y0 = int(round(y))
        x0 = int(round(x))
        y_lo = max(0, y0 - radius)
        y_hi = min(H, y0 + radius + 1)
        x_lo = max(0, x0 - radius)
        x_hi = min(W, x0 + radius + 1)
        mask[y_lo:y_hi, x_lo:x_hi] = False
    return mask


def build_atom_mask(shape, positions, radius):
    """모든 잠재 사이트를 마스크 (보수적)."""
    return _mask_positions(shape, positions, radius)


def build_label_masks(img_shape, positions, labels, radius, neighbor_radius=0):
    """
    Label 기반 per-frame 마스크.
    - bright (label==1) 사이트만 마스크 (empty 사이트는 유지: 유효 배경 샘플)
    - neighbor_radius > 0 이면 bright 사이트의 인접 사이트까지 추가 마스크 (PSF 오염 대비)

    positions : (n_sites, 2)
    labels    : (n_frames, n_sites) 0/1, 또는 (n_frames, ny, nx) 형태 가능
    returns   : bool array (n_frames, H, W)
    """
    H, W = img_shape
    labels = np.asarray(labels)
    if labels.ndim == 3:
        labels = labels.reshape(labels.shape[0], -1)
    if labels.ndim != 2 or labels.shape[1] != positions.shape[0]:
        raise ValueError(
            f"labels shape {labels.shape} incompatible with positions {positions.shape}"
        )
    n_frames = labels.shape[0]

    # 인접 사이트 계산용: 각 사이트의 nearest neighbors
    nbr_idx = None
    if neighbor_radius > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        # 각 사이트에서 반경 내 이웃 인덱스
        nbr_idx = tree.query_ball_point(positions, r=_median_spacing(positions) * neighbor_radius * 1.1)

    masks = np.ones((n_frames, H, W), dtype=bool)
    for f in range(n_frames):
        bright_sites = np.where(labels[f] > 0)[0]
        if bright_sites.size == 0:
            continue
        to_mask = set(bright_sites.tolist())
        if nbr_idx is not None:
            for s in bright_sites:
                to_mask.update(nbr_idx[s])
        pos_to_mask = positions[list(to_mask)]
        masks[f] = _mask_positions((H, W), pos_to_mask, radius)
    return masks


def _median_spacing(positions):
    """대략적인 격자 간격(px). 가장 가까운 이웃 거리의 중앙값."""
    from scipy.spatial import cKDTree
    if positions.shape[0] < 2:
        return 1.0
    tree = cKDTree(positions)
    d, _ = tree.query(positions, k=2)
    return float(np.median(d[:, 1]))


# ------------------------------------------------------------
# Main fitting routine
# ------------------------------------------------------------
def fit_lambda_from_atom_tif(
    tif_path,
    positions_path,
    bias_fixed,
    sigma_fixed,
    em_gain=300.0,
    sensitivity=4.15,
    mask_radius=3,
    bins=400,
    fit_bias=False,
    out_dir=None,
    tag=None,
    labels_path=None,
    neighbor_radius=0,
    fit_method='log',   # 'log' | 'linear_poisson'
):
    # 1) Load
    print(f"[load] TIF  : {tif_path}")
    img = imread(tif_path).astype(np.float32)
    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3:
        if img.shape[-1] <= 4 and img.shape[0] > 4:
            img = np.transpose(img, (2, 0, 1))
    N, H, W = img.shape
    print(f"         shape = {img.shape}")

    print(f"[load] pos  : {positions_path}")
    positions = np.load(positions_path)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"Unexpected positions shape: {positions.shape}")
    print(f"         n_sites = {positions.shape[0]}, radius = {mask_radius} px")

    # 2) Build mask
    if labels_path is not None:
        print(f"[load] lbl  : {labels_path}")
        labels = np.load(labels_path)
        print(f"         shape = {labels.shape}")
        masks_3d = build_label_masks(
            (H, W), positions, labels, mask_radius,
            neighbor_radius=neighbor_radius,
        )
        if masks_3d.shape[0] != N:
            raise ValueError(
                f"labels frame count {masks_3d.shape[0]} != tif frames {N}"
            )
        n_bg_total = int(masks_3d.sum())
        print(f"[mask] label-based. mean bg pixels/frame = {n_bg_total/N:.1f} / {H*W}")
        # apply per-frame mask
        data = img[masks_3d]
    else:
        mask_2d = build_atom_mask((H, W), positions, mask_radius)
        n_bg = int(mask_2d.sum())
        n_atom = H * W - n_bg
        print(f"[mask] conservative (all positions). bg/frame = {n_bg}/{H*W}  (atom-masked: {n_atom})")
        data = img[:, mask_2d].reshape(-1)

    print(f"[data] total bg samples = {data.size}")

    # 4) Histogram (Poisson 가중치)
    vmin, vmax = np.percentile(data, 0.01), np.percentile(data, 99.99)
    counts, bin_edges = np.histogram(data, bins=bins, range=(vmin, vmax), density=False)
    bin_width = bin_edges[1] - bin_edges[0]
    N_total = counts.sum()
    hist_y = counts / (N_total * bin_width)
    y_err = np.sqrt(np.maximum(counts, 1)) / (N_total * bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 5) Curve fit
    print(f"[fit ] σ fixed = {sigma_fixed:.4f}, em_gain = {em_gain}, sensitivity = {sensitivity}")
    print(f"       bias {'FREE' if fit_bias else f'fixed = {bias_fixed:.3f}'}")

    valid = counts > 0
    x_fit = bin_centers[valid]
    y_fit = hist_y[valid]
    s_fit = y_err[valid]

    if fit_bias:
        def model(x, bias, lam, scale):
            return basden_pdf(x, bias, sigma_fixed, lam, scale, em_gain, sensitivity)
        p0 = [bias_fixed, 0.02, 1.0]
        bounds = ([bias_fixed - 20, 1e-5, 0.0], [bias_fixed + 20, 2.0, np.inf])
    else:
        def model(x, lam, scale):
            return basden_pdf(x, bias_fixed, sigma_fixed, lam, scale, em_gain, sensitivity)
        p0 = [0.02, 1.0]
        bounds = ([1e-5, 0.0], [2.0, np.inf])

    if fit_method == 'log':
        # Log-space LSQ (EMCCD 관례, tail 무게)
        def log_model(x, *params):
            return np.log(np.maximum(model(x, *params), 1e-20))
        popt, pcov = curve_fit(
            log_model, x_fit, np.log(y_fit),
            p0=p0, bounds=bounds, maxfev=20000,
        )
    elif fit_method == 'linear_poisson':
        # Linear + Poisson weights (Pearson χ² approx)
        popt, pcov = curve_fit(
            model, x_fit, y_fit,
            p0=p0, sigma=s_fit, absolute_sigma=False,
            bounds=bounds, maxfev=20000,
        )
    else:
        raise ValueError(f"Unknown fit_method: {fit_method}")

    if fit_bias:
        bias_f, lam_f, scale_f = popt
    else:
        lam_f, scale_f = popt
        bias_f = bias_fixed

    # Bound 경고
    for name, val, lo, hi in [('lam', lam_f, bounds[0][-2], bounds[1][-2])]:
        rel_lo = (val - lo) / max(abs(hi - lo), 1e-12)
        rel_hi = (hi - val) / max(abs(hi - lo), 1e-12)
        if rel_lo < 1e-3 or rel_hi < 1e-3:
            print(f"  [WARN] {name}={val:.5f} at bound [{lo}, {hi}]")

    # 6) 결과 출력
    print("\n" + "=" * 50)
    print("   λ REFIT FROM ATOM IMAGES (stray light included)   ")
    print("=" * 50)
    print(f"Sensitivity  : {sensitivity:.3f} e-/ADU  (FIXED)")
    print(f"EM Gain      : {em_gain:.2f}  (FIXED)")
    print(f"Readout Sig  : {sigma_fixed:.3f} ADU  (FIXED)")
    print(f"Bias Offset  : {bias_f:.3f} ADU  ({'FIT' if fit_bias else 'FIXED'})")
    print(f"CIC+Stray λ  : {lam_f:.5f}  e-/pixel/frame   <-- refit")
    print("=" * 50)

    # 7) 시각화 (log + pull residual)
    y_model = model(bin_centers, *popt)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.semilogy(bin_centers, hist_y, 'o', color='gray', alpha=0.5, markersize=3, label='Bg pixels (atom-masked)')
    ax1.semilogy(bin_centers, y_model, 'r-', linewidth=2, label='New fit')
    y_gauss = scale_f * np.exp(-lam_f) / (np.sqrt(2 * np.pi) * sigma_fixed) \
              * np.exp(-0.5 * ((bin_centers - bias_f) / sigma_fixed) ** 2)
    ax1.semilogy(bin_centers, y_gauss, 'b--', linewidth=1, alpha=0.7, label='Readout (Gauss)')
    ax1.set_ylabel("PDF (log)")
    ax1.set_title(f"Basden fit on atom images  σ={sigma_fixed:.3f} (fixed), λ={lam_f:.4f}, bias={bias_f:.3f}")
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.2)

    resid = (hist_y - y_model) / np.maximum(y_err, 1e-12)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.plot(bin_centers, resid, 'k.', markersize=3)
    ax2.set_ylim(-10, 10)
    ax2.set_xlabel("Pixel Value (ADU)")
    ax2.set_ylabel("Pull")
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()

    out_dir_eff = out_dir if out_dir is not None else "."
    os.makedirs(out_dir_eff, exist_ok=True)
    tag_part = f"_{tag}" if tag else ""
    png_path = os.path.join(out_dir_eff, f"emccd_fit_atom{tag_part}.png")
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved  : {png_path}")

    result = {
        'tag': tag,
        'tif': os.path.abspath(tif_path),
        'positions': os.path.abspath(positions_path),
        'mask_radius': int(mask_radius),
        'n_bg_samples': int(data.size),
        'bias_offset': float(bias_f),
        'readout_sigma': float(sigma_fixed),
        'cic_lambda': float(lam_f),       # CIC + stray light 흡수
        'em_gain': float(em_gain),
        'sensitivity': float(sensitivity),
        'scale': float(scale_f),
        'fit_bias': bool(fit_bias),
    }
    json_path = os.path.join(out_dir_eff, f"emccd_fit_atom{tag_part}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Params saved: {json_path}")
    return result


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument("--tif", required=True, type=str,
                   help="실측 atom TIF (학습·추론 때 쓰인 조건과 동일한 데이터)")
    p.add_argument("--positions", required=True, type=str,
                   help="positions_data.npy (n_sites, 2) (y, x) pixel indices")
    p.add_argument("--sigma", required=True, type=float,
                   help="이전 dark-frame fit 에서 얻은 σ (ADU). 여기서는 고정.")
    p.add_argument("--bias", required=True, type=float,
                   help="이전 dark-frame fit 에서 얻은 bias (ADU). 기본은 고정.")
    p.add_argument("--em_gain", type=float, default=300.0)
    p.add_argument("--sensitivity", type=float, default=4.15)
    p.add_argument("--mask_radius", type=int, default=3,
                   help="원자 주변 마스크 반경 (px). 기본 3 → 7x7 박스 제외")
    p.add_argument("--labels", type=str, default=None,
                   help="(권장) test_labels*.npy. 있으면 각 frame에서 "
                        "bright atom만 제외. empty site는 배경 샘플로 활용.")
    p.add_argument("--neighbor_radius", type=int, default=0,
                   help="label mode에서 bright atom의 인접 사이트까지 추가 마스크 "
                        "(PSF 꼬리 오염 대비). 0=off, 1=즉시 이웃 포함.")
    p.add_argument("--fit_method", choices=['log', 'linear_poisson'], default='log',
                   help="log = log-space LSQ (EMCCD 관례, 논문 방식에 가까움). "
                        "linear_poisson = Pearson χ² (peak 중심).")
    p.add_argument("--bins", type=int, default=400)
    p.add_argument("--fit_bias", action='store_true',
                   help="bias 도 함께 fit (SDB ON 조건에서 bias 가 약간 shift 가능)")
    p.add_argument("--out_dir", type=str, default="fit_results_atom")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    fit_lambda_from_atom_tif(
        tif_path=args.tif,
        positions_path=args.positions,
        bias_fixed=args.bias,
        sigma_fixed=args.sigma,
        em_gain=args.em_gain,
        sensitivity=args.sensitivity,
        mask_radius=args.mask_radius,
        bins=args.bins,
        fit_bias=args.fit_bias,
        out_dir=args.out_dir,
        tag=args.tag,
        labels_path=args.labels,
        neighbor_radius=args.neighbor_radius,
        fit_method=args.fit_method,
    )
