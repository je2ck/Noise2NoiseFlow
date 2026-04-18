import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless: never pop up a window
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import i1
from tifffile import imread
import json
import os

# ==========================================
# 1. 물리 상수 설정 (Experimental Constants)
# ==========================================
# 앞선 분석에서 Pre-amp Gain x1.0 모드임이 밝혀졌습니다.
SENSITIVITY = 4.15  # e-/ADU (Sensitivity)
EM_GAIN = 300.0     # EM Gain (Fixed)


def basden_complete_model(x_adu, bias, sigma_adu, lam, scale):
    """
    Physically Correct Basden + Gaussian Model (Corrected)
    Gain and Sensitivity are fixed as global constants.
    """
    gain = EM_GAIN  # 고정값 사용
    
    # --- A. Gaussian Part (Readout Noise for 0 photons) ---
    # 0개 광자가 들어올 확률 = e^(-lambda)
    # 따라서 가우시안의 높이는 (1/sqrt(2pi*sigma)) * e^(-lambda) 가 되어야 함
    norm_gauss = 1.0 / (np.sqrt(2 * np.pi) * sigma_adu)
    
    prob_zero = np.exp(-lam) 
    gauss_pdf = prob_zero * norm_gauss * np.exp(-0.5 * ((x_adu - bias) / sigma_adu)**2)
    
    # --- B. Basden Part (Amplified Signal for >0 photons) ---
    # 1. 변수 변환: ADU -> Electron
    x_e = (x_adu - bias) * SENSITIVITY
    
    basden_pdf_e = np.zeros_like(x_adu)
    mask = x_e > 1e-2 
    
    if np.any(mask):
        z = x_e[mask]
        
        # P(x) = [sqrt(lam) / sqrt(gx)] * exp(...) * I1(...)
        
        term_coef = np.sqrt(lam) / (np.sqrt(gain * z) + 1e-12)
        term_exp = np.exp(-(z/gain + lam))
        arg_bessel = 2 * np.sqrt(lam * z / gain)
        
        val = term_coef * term_exp * i1(arg_bessel)
        
        # 3. Jacobian 변환 (Electron -> ADU)
        basden_pdf_e[mask] = val * SENSITIVITY 

    # --- C. Combine ---
    # Basden 식 자체에 이미 e^(-lam) 등 확률 정보가 포함되어 있음.
    # 따라서 여기서 'lam'을 곱하면 안 됩니다. (더하기만 해야 함)
    
    total_pdf = gauss_pdf + basden_pdf_e
    
    return scale * total_pdf

def analyze_background_noise(tiff_path, bins=400, fit_gain=False, out_dir=None, tag=None):
    print(f"Loading data from: {os.path.basename(tiff_path)}")

    # 1. 데이터 로드 및 전처리
    img = imread(tiff_path).astype(np.float32)
    data = img.reshape(-1)

    # Outlier 제거: 꼬리 0.01% 만 제거해 Basden 증폭 꼬리 보존
    vmin, vmax = np.percentile(data, 0.01), np.percentile(data, 99.99)
    # density=False 로 뽑은 뒤 Poisson 가중치 계산 후 density 로 변환
    counts, bin_edges = np.histogram(data, bins=bins, range=(vmin, vmax), density=False)
    bin_width = bin_edges[1] - bin_edges[0]
    N_total = counts.sum()
    hist_y = counts / (N_total * bin_width)                         # density
    y_err = np.sqrt(np.maximum(counts, 1)) / (N_total * bin_width)  # Poisson 오차 (density 단위)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 2. 초기값 추정 (Initial Guess)
    bias_init = bin_centers[np.argmax(hist_y)]
    # 가우시안 코어의 FWHM 으로 sigma 대략 추정 (하한 박힘 방지)
    peak_val = hist_y.max()
    above_half = bin_centers[hist_y > peak_val / 2]
    sigma_init = max(1.0, (above_half.max() - above_half.min()) / 2.355) if len(above_half) > 2 else 10.0
    lam_init = 0.05
    scale_init = 1.0

    print(f"Initial Guess: Bias={bias_init:.2f}, Sigma={sigma_init:.2f}")
    if fit_gain:
        print(f"Free params: Bias, Sigma, Lam, Scale, EM_Gain   Sensitivity={SENSITIVITY} (fixed)")
    else:
        print(f"Free params: Bias, Sigma, Lam, Scale            EM_Gain={EM_GAIN} (fixed), Sensitivity={SENSITIVITY} (fixed)")

    # 3. Fitting — log-space LSQ (EMCCD 관례. Meschede 2018 등 관련 논문과 정합.
    #    PDF 가 4~5 orders of magnitude 를 span 하므로 tail 가중치를 동등하게 주기 위함.
    #    σ 하한만 0.5 로 완화 (이전: 10.0 이 binding 되어 모든 조건에서 정확히 10.00 이
    #    반환되던 버그 수정).
    try:
        valid = counts > 0
        x_fit = bin_centers[valid]
        y_fit = hist_y[valid]

        if fit_gain:
            def model(x, b, s, l, sc, g):
                global EM_GAIN
                old = EM_GAIN
                EM_GAIN = g
                y = basden_complete_model(x, b, s, l, sc)
                EM_GAIN = old
                return y
            p0 = [bias_init, sigma_init, lam_init, scale_init, EM_GAIN]
            bounds = (
                [bias_init - 50, 0.5, 1e-5, 0.0,    50.0],
                [bias_init + 50, 200.0, 2.0, np.inf, 2000.0],
            )
        else:
            model = basden_complete_model
            p0 = [bias_init, sigma_init, lam_init, scale_init]
            bounds = (
                [bias_init - 50, 0.5,   1e-5, 0.0],
                [bias_init + 50, 200.0, 2.0,  np.inf],
            )

        def log_model(x, *params):
            return np.log(np.maximum(model(x, *params), 1e-20))

        popt, _ = curve_fit(
            log_model, x_fit, np.log(y_fit),
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )

        if fit_gain:
            bias_f, sigma_f, lam_f, scale_f, gain_f = popt
        else:
            bias_f, sigma_f, lam_f, scale_f = popt
            gain_f = EM_GAIN

        # 하한에 박혔는지 경고
        for name, val, lo, hi in [('bias', bias_f, bounds[0][0], bounds[1][0]),
                                  ('sigma', sigma_f, bounds[0][1], bounds[1][1]),
                                  ('lam',   lam_f,   bounds[0][2], bounds[1][2])]:
            rel_lo = (val - lo) / max(abs(hi - lo), 1e-12)
            rel_hi = (hi - val) / max(abs(hi - lo), 1e-12)
            if rel_lo < 1e-3 or rel_hi < 1e-3:
                print(f"  [WARN] {name}={val:.4f} is at the bound [{lo}, {hi}] — loosen and refit")
        
        # 4. 결과 출력
        print("\n" + "="*48)
        print("   PHYSICAL FITTING RESULTS   ")
        print("="*48)
        print(f"Sensitivity : {SENSITIVITY:.3f} e-/ADU (Fixed)")
        print(f"EM Gain     : {gain_f:.2f} {'(Fit)' if fit_gain else '(Fixed)'}")
        print(f"Bias Offset : {bias_f:.3f} ADU")
        print(f"Readout Sig : {sigma_f:.3f} ADU (-> {sigma_f*SENSITIVITY:.2f} e-)")
        print(f"CIC Lambda  : {lam_f:.5f} e-/pixel/frame")
        print("="*48)

        # 5. 시각화 (log + linear residual 2 panel)
        y_model = basden_complete_model(bin_centers, bias_f, sigma_f, lam_f, scale_f) \
                  if not fit_gain else model(bin_centers, *popt)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.semilogy(bin_centers, hist_y, 'o', color='gray', alpha=0.5, markersize=3, label='Data')
        ax1.semilogy(bin_centers, y_model, 'r-', linewidth=2, label=f'Best Fit')
        y_gauss = scale_f * np.exp(-lam_f) * (1.0/(np.sqrt(2*np.pi)*sigma_f)) \
                  * np.exp(-0.5*((bin_centers-bias_f)/sigma_f)**2)
        ax1.semilogy(bin_centers, y_gauss, 'b--', linewidth=1, alpha=0.7, label='Readout (Gauss)')
        ax1.set_ylabel("PDF (log)")
        ax1.set_title(f"EMCCD Noise Fit  bias={bias_f:.2f}, σ={sigma_f:.2f}, λ={lam_f:.4f}, gain={gain_f:.1f}")
        ax1.legend(); ax1.grid(True, which='both', alpha=0.2)

        # Pull residual
        resid = (hist_y - y_model) / np.maximum(y_err, 1e-12)
        ax2.axhline(0, color='k', lw=0.5)
        ax2.plot(bin_centers, resid, 'k.', markersize=3)
        ax2.set_ylim(-10, 10)
        ax2.set_xlabel("Pixel Value (ADU)")
        ax2.set_ylabel("Pull (resid / σ)")
        ax2.grid(True, alpha=0.2)
        plt.tight_layout()

        out_dir_eff = out_dir if out_dir is not None else "."
        os.makedirs(out_dir_eff, exist_ok=True)
        tag_part = f"_{tag}" if tag else ""
        save_path = os.path.join(out_dir_eff, f"emccd_fit{tag_part}.png")
        plt.savefig(save_path, dpi=200)
        print(f"Plot saved to: {save_path}")
        plt.close(fig)

        result = {
            'tag':            tag,
            'tiff':           os.path.abspath(tiff_path),
            'bias_offset':    float(bias_f),
            'readout_sigma':  float(sigma_f),
            'cic_lambda':     float(lam_f),
            'em_gain':        float(gain_f),
            'sensitivity':    float(SENSITIVITY),
            'scale':          float(scale_f),
            'fit_gain':       bool(fit_gain),
            'n_pixels':       int(N_total),
        }
        json_path = os.path.join(out_dir_eff, f"emccd_fit{tag_part}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Params saved to: {json_path}")
        return result

    except Exception as e:
        print(f"Fitting Failed: {e}")
        return None

# ==========================================
# 실행 부
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit Basden EMCCD noise model to background TIFF data")
    parser.add_argument("tiff", type=str, help="Path to background TIFF file (pure dark frames)")
    parser.add_argument("--bins", type=int, default=400)
    parser.add_argument("--fit_gain", action='store_true',
                        help="EM gain 도 피팅 파라미터로 (기본: 고정 300)")
    parser.add_argument("--sensitivity", type=float, default=None,
                        help="기본 4.15 (train/infer 와 일치). 바꾸고 싶을 때만 지정")
    parser.add_argument("--em_gain", type=float, default=None,
                        help="EM gain 초기/고정값. 기본 300")
    parser.add_argument("--out_dir", type=str, default=".", help="결과 저장 디렉토리")
    parser.add_argument("--tag", type=str, default=None, help="파일명 suffix (예: 5ms)")
    args = parser.parse_args()

    if args.sensitivity is not None:
        SENSITIVITY = args.sensitivity
    if args.em_gain is not None:
        EM_GAIN = args.em_gain

    analyze_background_noise(args.tiff, bins=args.bins, fit_gain=args.fit_gain,
                             out_dir=args.out_dir, tag=args.tag)