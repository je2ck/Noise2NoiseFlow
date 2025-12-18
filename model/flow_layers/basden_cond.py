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
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1, device=device))
        
        # 2. Bias (Offset 보정용)
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, device=device))
    
    def _forward_and_log_det_jacobian(self, x, **kwargs):
        scale = torch.exp(self.log_scale)
        z = x * scale + self.bias
        log_det = self.log_scale.sum() * x.shape[2] * x.shape[3]
        return z, log_det.expand(x.shape[0])

    def _inverse(self, z, **kwargs):
        scale = torch.exp(self.log_scale)
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

        # 정규화 파라미터
        self.norm_min = float(config['vmin'])
        self.norm_max = float(config['vmax'])

        # --- 2. Look-up Table 생성 (Dark Reference CDF) ---
        # 이 테이블은 "신호가 0일 때(Dark)"의 분포를 나타냅니다.
        x_min = self.bias - 6 * self.sigma
        x_max = float(max_adu)
        
        self.x_grid_np = np.linspace(x_min, x_max, num_bins).astype(np.float32)
        
        # (1) 0 Photon Part (Gaussian Only)
        p_zero = np.exp(-self.lam)
        pdf_zero = p_zero * norm.pdf(self.x_grid_np, loc=self.bias, scale=self.sigma)
        
        # (2) >0 Photon Part (Basden Formula for Dark/CIC)
        x_e = (self.x_grid_np - self.bias) * self.sens
        pdf_signal = np.zeros_like(self.x_grid_np)
        mask = x_e > 1e-4
        
        if np.any(mask):
            z = x_e[mask]
            g = self.real_gain
            l = self.lam
            
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
        
        cdf_vals = np.cumsum(pdf_vals) * dx
        cdf_vals = np.clip(cdf_vals, 1e-7, 1.0 - 1e-7)

        # Tensor 등록
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

    def _get_signal_dependent_scale(self, clean_norm):
        """Clean 이미지를 기반으로 노이즈 스케일링 팩터를 계산합니다."""
        # 1. Denormalize Clean Image (0~1 -> ADU)
        scale_norm = (self.norm_max - self.norm_min)
        clean_adu = clean_norm * scale_norm + self.norm_min
        
        # 2. Signal 추출 (Bias 제거)
        signal_adu = torch.clamp(clean_adu - self.bias, min=0.0)
        
        # 3. 물리적 분산 계산 (EMCCD Shot Noise)
        # Var_total = 2 * Gain * Signal + Readout^2
        # (Excess Noise Factor F^2 ~ 2 for EMCCD)
        var_shot = 2.0 * self.real_gain * signal_adu
        var_read = self.sigma ** 2
        
        sigma_total = torch.sqrt(var_shot + var_read)
        sigma_dark = self.sigma  # Reference (Dark) Sigma
        
        # 4. 스케일링 비율 (밝을수록 sigma_total이 커지므로 ratio는 작아짐)
        # 이 비율로 Residual을 나눠서 Dark CDF 폭에 맞춥니다.
        ratio = sigma_dark / (sigma_total + 1e-8)
        
        return ratio

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        # x: Normalized Residual (Input)
        norm_scale = (self.norm_max - self.norm_min)
        
        # 1. Clean 이미지 가져오기 및 스케일 팩터 계산
        clean = kwargs.get('clean', torch.zeros_like(x))
        scale_factor = self._get_signal_dependent_scale(clean) # (N, C, H, W)
        
        # 2. Denormalize Residual (0~1 -> ADU)
        # 입력 x는 residual이므로 단순히 스케일만 복원합니다.
        # 중심점 이동은 나중에 Bias를 더해서 수행합니다.
        x_res_adu = x * norm_scale
        
        # 3. Signal-Dependent Scaling (핵심!)
        # 밝은 곳의 큰 잔차를 작게 줄여서 Dark CDF에 맞춤
        x_scaled_res = x_res_adu * scale_factor
        
        # 4. Bias 더하기 (Dark CDF의 중심에 맞춤)
        x_input_cdf = x_scaled_res + self.bias
        
        # 안전장치 & Clamp
        if x_input_cdf.dtype == torch.float64:
            x_input_cdf = x_input_cdf.float()
        x_clamped = torch.clamp(x_input_cdf, min=self.x_grid.min(), max=self.x_grid.max())

        # 5. CDF Transform (Dark Table 사용)
        u = self._interp(x_clamped, self.x_grid, self.cdf_table)
        u = torch.clamp(u, 1e-5, 1.0 - 1e-5)
        z = torch.erfinv(2 * u - 1) * np.sqrt(2)
        
        # 6. Log-Det Calculation
        # (1) Basden Transform part
        p_basden = self._interp(x_clamped, self.x_grid, self.pdf_table)
        dlog_p_basden = torch.log(p_basden + 1e-8)
        log_p_gauss = -0.5 * (z ** 2) - 0.5 * np.log(2 * np.pi)
        log_det_physics = dlog_p_basden - log_p_gauss
        
        # (2) Scaling part (Residual Scale + Signal Scale)
        # x -> x_res_adu (norm_scale)
        # x_res_adu -> x_scaled_res (scale_factor)
        log_det_norm = torch.log(torch.abs(torch.tensor(norm_scale, device=x.device)) + 1e-8)
        log_det_signal = torch.log(scale_factor + 1e-8)
        
        total_dlogdet = log_det_physics + log_det_norm + log_det_signal
        
        return z, total_dlogdet.sum(dim=[1, 2, 3])

    def _inverse(self, z, **kwargs):
        norm_scale = (self.norm_max - self.norm_min)
        
        # Clean 이미지 기반 스케일 복원
        clean = kwargs.get('clean', torch.zeros_like(z))
        scale_factor = self._get_signal_dependent_scale(clean)
        
        # 1. z -> u (Normal CDF)
        u = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        
        # 2. u -> x (Inverse Dark CDF)
        x_input_cdf = self._interp(u, self.cdf_table, self.x_grid)
        
        # 3. Bias 제거 (to Residual)
        x_scaled_res = x_input_cdf - self.bias
        
        # 4. Signal Scale 복원 (나눴던 것 다시 곱하기)
        x_res_adu = x_scaled_res / (scale_factor + 1e-8)
        
        # 5. Renormalize (ADU -> 0~1)
        x_norm = x_res_adu / (norm_scale + 1e-8)
        
        return x_norm