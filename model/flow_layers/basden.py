import torch
import torch.nn as nn
import numpy as np
from scipy.special import i1
from scipy.stats import norm

class BasdenAdaptor(nn.Module):
    def __init__(self, num_channels, device='cuda'):
        super().__init__()
        self.device = device
        self.name = "BasdenAdaptor"
        
        # 1. Scale (Gain/Sigma 보정용)
        # 초기값 0 -> exp(0)=1.0 (변화 없음에서 시작)
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1, device=device))
        
        # 2. Bias (Offset 보정용)
        # 초기값 0 (변화 없음에서 시작)
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, device=device))
    
    def _forward_and_log_det_jacobian(self, x, **kwargs):
        # x: Normalized input (0~1)
        
        scale = torch.exp(self.log_scale)
        
        # 선형 변환: z = scale * x + bias
        z = x * scale + self.bias
        
        # Log-Det Jacobian: log(scale) * H * W
        # 픽셀마다 scale을 곱해주므로, 전체 부피 변화는 scale의 합입니다.
        log_det = self.log_scale.sum() * x.shape[2] * x.shape[3]
        
        return z, log_det.expand(x.shape[0])

    def _inverse(self, z, **kwargs):
        scale = torch.exp(self.log_scale)
        
        # 역변환: x = (z - bias) / scale
        x = (z - self.bias) / (scale + 1e-8)
        return x
    
    
class BasdenFlowLayer(nn.Module):
    def __init__(self, config, device='cuda', num_bins=30000, max_adu=65535):
        super().__init__()
        self.device = device
        self.name = "BasdenPhysicalLayer"
        
        # --- 1. 물리 파라미터 로드 ---
        self.bias = float(config['bias_offset'])
        self.sigma = float(config['readout_sigma']) 
        
        self.real_gain = float(config['em_gain'])
        self.sens = float(config['sensitivity'])
        self.lam = float(config['cic_lambda'])

        # [수정 1] 정규화 파라미터를 __init__에서 미리 저장 (forward에서 쓰기 위해)
        self.norm_min = float(config['vmin'])
        self.norm_max = float(config['vmax'])

        # --- 2. Look-up Table 생성 (Correct Basden PDF) ---
        x_min = self.bias - 6 * self.sigma # [권장] 꼬리를 조금 더 넉넉하게 (5 -> 6 sigma)
        x_max = float(max_adu)
        
        # [권장] num_bins 20000 -> 30000 (해상도 증가)
        self.x_grid_np = np.linspace(x_min, x_max, num_bins).astype(np.float32)
        
        # (1) 0 Photon Part (Gaussian Only)
        p_zero = np.exp(-self.lam)
        pdf_zero = p_zero * norm.pdf(self.x_grid_np, loc=self.bias, scale=self.sigma)
        
        # (2) >0 Photon Part (Basden Formula)
        x_e = (self.x_grid_np - self.bias) * self.sens
        
        pdf_signal = np.zeros_like(self.x_grid_np)
        mask = x_e > 1e-4
        
        if np.any(mask):
            z = x_e[mask]
            g = self.real_gain
            l = self.lam
            
            # Basden Formula (Correct logic maintained)
            term_coef = np.sqrt(l) / (np.sqrt(g * z) + 1e-12)
            term_exp = np.exp(-(z/g + l))
            arg_bessel = 2 * np.sqrt(l * z / g)
            
            val_e = term_coef * term_exp * i1(arg_bessel)
            pdf_signal[mask] = val_e * self.sens

        # (3) Combine & Normalize
        pdf_vals = pdf_zero + pdf_signal
        
        dx = self.x_grid_np[1] - self.x_grid_np[0]
        area = np.sum(pdf_vals) * dx
        pdf_vals /= area 
        
        # CDF 계산
        cdf_vals = np.cumsum(pdf_vals) * dx
        cdf_vals = np.clip(cdf_vals, 1e-7, 1.0 - 1e-7)

        # --- 3. Tensor 등록 (float32 변환 유지) ---
        self.x_grid = torch.from_numpy(self.x_grid_np.astype(np.float32)).to(device)
        self.pdf_table = torch.from_numpy(pdf_vals.astype(np.float32)).to(device)
        self.cdf_table = torch.from_numpy(cdf_vals.astype(np.float32)).to(device)

    def _interp(self, x, x_data, y_data):
        indices = torch.searchsorted(x_data, x)
        indices = torch.clamp(indices, 1, len(x_data) - 1)
        x0, x1 = x_data[indices - 1], x_data[indices]
        y0, y1 = y_data[indices - 1], y_data[indices]
        slope = (y1 - y0) / (x1 - x0 + 1e-8)
        return y0 + slope * (x - x0)

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        # [수정 2] __init__에서 저장한 self.norm_min/max 사용 (오타 수정됨)
        norm_min = self.norm_min
        norm_max = self.norm_max
        
        scale = (norm_max - norm_min)
        
        # 1. Denormalize: (0~1) -> Real ADU
        x_adu = x * scale + norm_min
        
        # 안전장치
        if x_adu.dtype == torch.float64:
            x_adu = x_adu.float()
            
        x_clamped = torch.clamp(x_adu, min=self.x_grid.min(), max=self.x_grid.max())

        # 2. x_adu -> u -> z
        u = self._interp(x_clamped, self.x_grid, self.cdf_table)
        u = torch.clamp(u, 1e-5, 1.0 - 1e-5)
        z = torch.erfinv(2 * u - 1) * np.sqrt(2)
        
        # 3. Log-Det Calculation
        p_basden = self._interp(x_clamped, self.x_grid, self.pdf_table)
        dlog_p_basden = torch.log(p_basden + 1e-8)
        log_p_gauss = -0.5 * (z ** 2) - 0.5 * np.log(2 * np.pi)
        
        log_det_physics = dlog_p_basden - log_p_gauss
        log_det_norm = torch.log(torch.abs(torch.tensor(scale, device=x.device)) + 1e-8)
        
        total_dlogdet = log_det_physics + log_det_norm
        
        if torch.isnan(total_dlogdet).any():
             # print(f"NaN Detected!") # 필요시 주석 해제
             pass

        return z, total_dlogdet.sum(dim=[1, 2, 3])

    def _inverse(self, z, **kwargs):
        # 1. z -> u (Normal CDF)
        u = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        
        # 2. u -> x_adu (Inverse Basden CDF)
        x_adu = self._interp(u, self.cdf_table, self.x_grid)
        
        # [수정 3] Renormalize: Real ADU -> (0~1)
        # 생성된 이미지는 다시 0~1 사이 값이어야 함
        norm_min = self.norm_min
        norm_max = self.norm_max
        scale = (norm_max - norm_min)
        
        x_norm = (x_adu - norm_min) / (scale + 1e-8)
        
        return x_norm