import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.special import i1
from tifffile import imread

def basden_model_func(x, bias, sigma, gain, lam, scale):
    """
    Fitting을 위한 Basden + Gaussian 혼합 모델 함수
    x: ADU 값
    bias: 오프셋
    sigma: Readout Noise Std
    gain: EM Gain
    lam: CIC Rate (평균 광자 수)
    scale: 히스토그램 높이 보정용
    """
    # 1. Zero-photon Peak (Gaussian)
    # P(0) approx exp(-lam). 
    # 하지만 Fitting을 위해선 이 비율도 자유도로 두는 게 더 잘 맞습니다.
    # 여기서는 근사적으로 Gaussian 항과 Basden 항을 더합니다.
    
    # Gaussian Part (Main Peak)
    gauss = np.exp(-0.5 * ((x - bias) / sigma)**2)
    # 정규화 상수는 scale에 포함시킨다고 가정
    
    # Basden Part (Tail)
    # x > bias 인 영역에서만 유효
    x_shifted = x - bias
    
    # Bessel 항 계산 (Overflow 방지를 위해 clip)
    z = x_shifted
    # z가 0 이하거나 너무 작으면 Basden 값은 0
    # 계산 안정성을 위해 마스킹
    basden = np.zeros_like(x)
    
    mask = z > 1e-1  # 0보다 조금 큰 값 이상
    z_m = z[mask]
    
    # Basden Formula: sqrt(lam / (g*x)) * exp(...) * I1(...)
    # 수식: P(x) ~ (1/x) * exp(-x/g) 형태가 지배적임 (CIC의 경우)
    # 여기서는 친구분이 언급한 Bessel 식을 그대로 씁니다.
    
    arg = 2 * np.sqrt(lam * z_m / gain)
    val = (np.exp(-(z_m/gain + lam)) / np.sqrt(z_m * gain * lam)) * i1(arg)
    
    # Basden 항의 비중은 lam(CIC rate)에 비례
    basden[mask] = val * lam  
    
    # 최종 모델: (Gaussian * (1-lam)) + (Basden * lam) 형태의 스케일링
    # Fitting 편의상 A * Gauss + B * Basden 꼴로 단순화
    return scale * ((1-lam)*gauss + 500 * basden) 
    # 500은 Basden 값이 너무 작아서 fitting 시 무시되는 걸 막기 위한 가중치 (나중에 scale로 보정됨)

def fit_real_data(tiff_path):
    print(f"Loading {tiff_path}...")
    img = imread(tiff_path).astype(np.float32)
    data = img.reshape(-1)
    
    # 1. 히스토그램 생성 (Fitting의 목표물)
    # 데이터 범위를 보고 적절히 자릅니다 (너무 먼 Outlier 제외)
    vmin, vmax = np.percentile(data, 0.1), np.percentile(data, 99.9)
    # 꼬리를 잘 보기 위해 bin을 넉넉히 잡습니다.
    hist_y, bin_edges = np.histogram(data, bins=300, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # 2. 초기값 추정 (Initial Guess) - 이게 중요함!
    # Bias: 데이터의 최빈값(Mode)
    bias_init = bin_centers[np.argmax(hist_y)]
    
    # Sigma: FWHM 등을 이용해서 대충 추정 (보통 10~50 사이)
    sigma_init = 20.0 
    
    # Gain: 실험 셋팅값 (예: 300)
    gain_init = 1000.0 
    
    # Lambda (CIC): 보통 0.01 ~ 0.1 사이 (매우 작음)
    lam_init = 0.02
    
    # Scale: 히스토그램 최대값
    scale_init = np.max(hist_y)

    print(f"Initial Guesses: Bias={bias_init:.1f}, Sig={sigma_init}, Gain={gain_init}, Lam={lam_init}")
    
    def log_basden_model(x, bias, sigma, gain, lam, scale):
        # 모델 값에 log를 취해서 반환 (0 방지 위해 clip)
        val = basden_model_func(x, bias, sigma, gain, lam, scale)
        return np.log(np.maximum(val, 1e-20))

    try:
        # 실제 데이터에도 log를 취함 (0인 빈은 제외)
        valid_mask = hist_y > 0
        x_data = bin_centers[valid_mask]
        y_data_log = np.log(hist_y[valid_mask])
        
        popt, pcov = curve_fit(
            log_basden_model, 
            x_data, 
            y_data_log, 
            p0=[bias_init, sigma_init, gain_init, lam_init, scale_init],
            bounds=(
                [bias_init-50, 1.0, 10.0, 0.0001, 0], # Lower bound
                [bias_init+50, 200.0, 5000.0, 1.0, np.inf] # Upper bound
            )
        )

        bias_fit, sigma_fit, gain_fit, lam_fit, scale_fit = popt
        print("\n=== Fitting Results ===")
        print(f"Bias Offset : {bias_fit:.2f}")
        print(f"Readout Sig : {sigma_fit:.2f}")
        print(f"EM Gain     : {gain_fit:.2f}")
        print(f"CIC Lambda  : {lam_fit:.4f}")
        print("=======================")
        
        # 4. 결과 시각화
        plt.figure(figsize=(10, 6))
        plt.semilogy(bin_centers, hist_y, 'k.', label='Real Data', alpha=0.5)
        plt.semilogy(bin_centers, basden_model_func(bin_centers, *popt), 'r-', label='Fitted Basden Model')
        plt.ylim(bottom=1e-6)
        plt.title(f"Parameter Fitting Result\nGain={gain_fit:.0f}, CIC={lam_fit:.3f}, Sig={sigma_fit:.1f}")
        plt.xlabel("ADU")
        plt.ylabel("Probability (Log)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("fitting_result.png")
        plt.show()
        
        return bias_fit, sigma_fit, gain_fit, lam_fit

    except Exception as e:
        print(f"Fitting failed: {e}")
        return bias_init, sigma_init, gain_init, lam_init

# 실행
# tiff_path를 본인 경로로 수정하세요
# bias, sigma, gain, lam = fit_real_data("./data_atom/background.tif")
if __name__ == "__main__":
    bg_stack_path = "./data_atom/data_atom_8_10_20_conseq/background.tif"
    fit_real_data(bg_stack_path)