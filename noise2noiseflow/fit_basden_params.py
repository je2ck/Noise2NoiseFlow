import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import i1
from tifffile import imread
import os

# ==========================================
# 1. 물리 상수 설정 (Experimental Constants)
# ==========================================
# 앞선 분석에서 Pre-amp Gain x1.0 모드임이 밝혀졌습니다.
SENSITIVITY = 4.88  # e-/ADU (Sensitivity)


def basden_complete_model(x_adu, bias, sigma_adu, gain, lam, scale):
    """
    Physically Correct Basden + Gaussian Model (Corrected)
    """
    
    # --- A. Gaussian Part (Readout Noise for 0 photons) ---
    # 0개 광자가 들어올 확률 = e^(-lambda)
    # 따라서 가우시안의 높이는 (1/sqrt(2pi*sigma)) * e^(-lambda) 가 되어야 함
    norm_gauss = 1.0 / (np.sqrt(2 * np.pi) * sigma_adu)
    
    # [수정 1] 가중치를 (1-lam) 근사식 대신 정확한 exp(-lam) 사용
    prob_zero = np.exp(-lam) 
    gauss_pdf = prob_zero * norm_gauss * np.exp(-0.5 * ((x_adu - bias) / sigma_adu)**2)
    
    # --- B. Basden Part (Amplified Signal for >0 photons) ---
    # 1. 변수 변환: ADU -> Electron
    x_e = (x_adu - bias) * SENSITIVITY
    
    basden_pdf_e = np.zeros_like(x_adu)
    mask = x_e > 1e-2 
    
    if np.any(mask):
        z = x_e[mask]
        
        # [수정 2] Basden Formula 수식 교정
        # 이전: denom에 lam이 들어감 (오류) -> denom = np.sqrt(gain * z * lam)
        # 수정: 분자에 sqrt(lam)을 곱하고, 분모에서는 뺌
        
        # P(x) = [sqrt(lam) / sqrt(gx)] * exp(...) * I1(...)
        
        term_coef = np.sqrt(lam) / (np.sqrt(gain * z) + 1e-12)
        term_exp = np.exp(-(z/gain + lam))
        arg_bessel = 2 * np.sqrt(lam * z / gain)
        
        val = term_coef * term_exp * i1(arg_bessel)
        
        # 3. Jacobian 변환 (Electron -> ADU)
        basden_pdf_e[mask] = val * SENSITIVITY 

    # --- C. Combine ---
    # [수정 3] Basden 식 자체에 이미 e^(-lam) 등 확률 정보가 포함되어 있음.
    # 따라서 여기서 'lam'을 곱하면 안 됩니다. (더하기만 해야 함)
    
    total_pdf = gauss_pdf + basden_pdf_e
    
    return scale * total_pdf

def analyze_background_noise(tiff_path):
    print(f"Loading data from: {os.path.basename(tiff_path)}")
    
    # 1. 데이터 로드 및 전처리
    img = imread(tiff_path).astype(np.float32)
    data = img.reshape(-1)
    
    # Outlier 제거 (Fitting 안정성)
    vmin, vmax = np.percentile(data, 0.1), np.percentile(data, 99.9)
    # 히스토그램 생성 (Density=True 필수)
    hist_y, bin_edges = np.histogram(data, bins=200, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # 2. 초기값 추정 (Initial Guess)
    bias_init = bin_centers[np.argmax(hist_y)]  # 최빈값
    sigma_init = 25.0   # 이전 분석 결과 반영
    gain_init = 285.0   # 이전 분석 결과 반영
    lam_init = 0.05     # 정상적인 CIC 예상값
    scale_init = 1.0    # Density=True이므로 1에 가까움 (하지만 자유도로 둠)
    
    print(f"Initial Guess: Bias={bias_init:.1f}, Sig={sigma_init}, Gain={gain_init}")

    # 3. Fitting (Log Space)
    def log_objective(x, b, s, g, l, sc):
        # Log fitting을 위해 모델값에 log를 취함
        model_val = basden_complete_model(x, b, s, g, l, sc)
        return np.log(np.maximum(model_val, 1e-20)) # log(0) 방지
    
    try:
        # 데이터가 있는 빈만 선택
        valid = hist_y > 0
        x_fit = bin_centers[valid]
        y_fit = np.log(hist_y[valid])
        
        # Bounds 설정 (물리적 의미에 맞게)
        popt, pcov = curve_fit(
            log_objective, x_fit, y_fit,
            p0=[bias_init, sigma_init, gain_init, lam_init, scale_init],
            bounds=(
                # Lower: Bias, Sig, Gain, Lam, Scale
                [bias_init-50, 10.0, 200.0, 1e-5, 0],   
                # Upper: Bias, Sig, Gain, Lam, Scale
                [bias_init+50, 100.0, 500.0, 1.0, np.inf]
            )
        )
        
        bias_f, sigma_f, gain_f, lam_f, scale_f = popt
        
        # 4. 결과 출력
        print("\n" + "="*40)
        print("   PHYSCIAL FITTING RESULTS (Final)   ")
        print("="*40)
        print(f"Sensitivity : {SENSITIVITY} e-/ADU (Fixed)")
        print(f"Bias Offset : {bias_f:.2f} ADU")
        print(f"Readout Sig : {sigma_f:.2f} ADU (-> {sigma_f*SENSITIVITY:.1f} e-)")
        print(f"EM Gain     : {gain_f:.2f}")
        print(f"CIC Lambda  : {lam_f:.4f} e-/pixel/frame")
        print("="*40)
        
        # 5. 시각화
        plt.figure(figsize=(10, 6))
        
        # Data
        plt.semilogy(bin_centers, hist_y, 'o', color='gray', alpha=0.5, markersize=4, label='Data')
        
        # Total Fit
        y_model = basden_complete_model(bin_centers, *popt)
        plt.semilogy(bin_centers, y_model, 'r-', linewidth=2, label=f'Best Fit (CIC={lam_f:.4f})')
        
        # Components (분리해서 보여주기)
        y_gauss = scale_f * (1-lam_f) * (1.0/(np.sqrt(2*np.pi)*sigma_f)) * np.exp(-0.5*((bin_centers-bias_f)/sigma_f)**2)
        plt.semilogy(bin_centers, y_gauss, 'b--', linewidth=1, alpha=0.7, label='Readout Noise (Gauss)')
        
        plt.title(f"EMCCD Noise Analysis\nGain={gain_f:.1f}, $\sigma_{{read}}$={sigma_f:.1f}, $\lambda_{{CIC}}$={lam_f:.4f}")
        plt.xlabel("Pixel Value (ADU)")
        plt.ylabel("Probability Density (Log scale)")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        
        save_path = "emccd_fitting_final.png"
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
        plt.show()
        
        return bias_f, sigma_f, gain_f, lam_f

    except Exception as e:
        print(f"Fitting Failed: {e}")
        return None

# ==========================================
# 실행 부
# ==========================================
if __name__ == "__main__":
    # 경로를 본인의 데이터 경로로 수정하세요
    file_path = "./data_atom/data_atom_8_10_20_conseq/background.tif"
    analyze_background_noise(file_path)