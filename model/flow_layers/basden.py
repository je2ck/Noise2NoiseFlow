import torch
import torch.nn as nn
import numpy as np
from scipy.special import i1 # modified besself function of first kind, order 1 
from scipy.stats import norm

class BasdenFlowLayer(nn.Module):
    def __init__(self, config, device='cuda', num_bins=20000, max_adu=65535):
        super().__init__()
        self.device = device
        self.name = "BasdenPhysicalLayer"
        
        # --- 1. 물리 파라미터 로드 ---
        self.bias = config['bias_offset']
        self.sigma = config['readout_sigma'] # ADU 단위
        
        # [중요] Gain과 Sensitivity의 단위 명확화
        # config.em_gain: 순수 증폭 배율 (예: 300)
        # config.sensitivity: e-/ADU (예: 4.88)
        self.real_gain = config['em_gain']
        self.sens = config['sensitivity']
        self.lam = config['cic_lambda']

        # --- 2. Look-up Table 생성 (Correct Basden PDF) ---
        # 꼬리 부분(Tail)까지 충분히 커버하기 위해 x_grid를 촘촘하게 설정
        x_min = self.bias - 5 * self.sigma
        x_max = max_adu 
        
        # x_grid: ADU 단위
        self.x_grid_np = np.linspace(x_min, x_max, num_bins).astype(np.float32)
        
        # (1) 0 Photon Part (Gaussian Only)
        # P(0) = exp(-lam)
        p_zero = np.exp(-self.lam)
        pdf_zero = p_zero * norm.pdf(self.x_grid_np, loc=self.bias, scale=self.sigma)
        
        # (2) >0 Photon Part (Basden Formula)
        # 변수 변환: ADU -> Electrons
        # x_e = (ADU - Bias) * Sensitivity
        x_e = (self.x_grid_np - self.bias) * self.sens
        
        pdf_signal = np.zeros_like(self.x_grid_np)
        
        # Basden 식 적용 (x_e > 0 인 구간만)
        mask = x_e > 1e-4
        if np.any(mask):
            z = x_e[mask]
            g = self.real_gain
            l = self.lam
            
            # 앞서 우리가 수정한 완벽한 Basden 식
            # P_e(z) 계산
            term_coef = np.sqrt(l) / (np.sqrt(g * z) + 1e-12)
            term_exp = np.exp(-(z/g + l))
            arg_bessel = 2 * np.sqrt(l * z / g)
            
            val_e = term_coef * term_exp * i1(arg_bessel)
            
            # Jacobian 변환: P_adu(x) = P_e(z) * |dz/dx| = P_e(z) * Sensitivity
            pdf_signal[mask] = val_e * self.sens

        # (3) Combine & Normalize
        # Basden 식 자체에 (1-exp(-lam)) 확률이 포함되어 있지 않으므로 
        # 그냥 더해주면 됩니다. (수식상 자연스럽게 연결됨)
        # 하지만 수치적 오차 보정을 위해 전체 합으로 한번 나눠줍니다.
        
        pdf_vals = pdf_zero + pdf_signal
        
        dx = self.x_grid_np[1] - self.x_grid_np[0]
        area = np.sum(pdf_vals) * dx
        pdf_vals /= area  # 정확히 1로 정규화
        
        # CDF 계산
        cdf_vals = np.cumsum(pdf_vals) * dx
        cdf_vals = np.clip(cdf_vals, 1e-6, 1.0 - 1e-6)

        # --- 3. Tensor 등록 ---
        self.x_grid = torch.from_numpy(self.x_grid_np.astype(np.float32)).to(device)
        self.pdf_table = torch.from_numpy(pdf_vals.astype(np.float32)).to(device)
        self.cdf_table = torch.from_numpy(cdf_vals.astype(np.float32)).to(device)

    def _interp(self, x, x_data, y_data):
        """PyTorch 1D Linear Interpolation"""
        indices = torch.searchsorted(x_data, x)
        indices = torch.clamp(indices, 1, len(x_data) - 1)
        
        x0, x1 = x_data[indices - 1], x_data[indices]
        y0, y1 = y_data[indices - 1], y_data[indices]
        
        slope = (y1 - y0) / (x1 - x0 + 1e-8)
        return y0 + slope * (x - x0)

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        # x: Input ADU Image
        
        # 1. x -> u (Basden CDF)
        u = self._interp(x, self.x_grid, self.cdf_table)
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        
        # 2. u -> z (Inverse Normal CDF, Gaussian)
        z = torch.erfinv(2 * u - 1) * np.sqrt(2)
        
        # 3. Log-Det Jacobian
        p_basden = self._interp(x, self.x_grid, self.pdf_table)
        log_p_gauss = -0.5 * (z ** 2) - 0.5 * np.log(2 * np.pi)
        
        dlogdet = torch.log(p_basden + 1e-10) - log_p_gauss
        
        return z, dlogdet.sum(dim=[1, 2, 3])

    def _inverse(self, z, **kwargs):
        # 1. z -> u (Normal CDF)
        u = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        
        # 2. u -> x (Inverse Basden CDF)
        # CDF는 단조증가 함수이므로 x와 y의 역할을 바꿔서 인터폴레이션 하면 역함수가 됨
        # x_data = self.cdf_table, y_data = self.x_grid
        x = self._interp(u, self.cdf_table, self.x_grid)
        
        return x