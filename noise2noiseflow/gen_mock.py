import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import torch
from tifffile import imread
import os
import sys
import types
from scipy.stats import norm, poisson
from scipy import fftpack

# --- [Path Setup] ---
sys.path.append("../") 
from model.noise2noise_flow import Noise2NoiseFlow
from train_atom import init_params, _ensure_channels, GLOBAL_VMIN, GLOBAL_VMAX

# CUDA Hack
if not torch.cuda.is_available():
    def _fake_cuda(self, device=None, non_blocking=False): return self
    torch.Tensor.cuda = _fake_cuda
    torch.nn.Module.cuda = _fake_cuda

# --- 1. System Configuration (수정됨) ---
class SystemConfig:
    def __init__(self):
        # Physics
        self.R_sc = 1.69e5
        self.NA = 0.5
        self.wavelength = 556e-9
        self.pixel_size_um = 13.0
        self.magnification = 30.0
        self.T_optics = 0.3
        self.QE = 0.9
        
        # self.eta_geo = (1 - np.sqrt(1 - self.NA**2)) / 2
        self.eta_geo = 0.08  # 고정된 값으로 설정
        self.eta_total = self.eta_geo * self.T_optics * self.QE
        
        res_object_plane = 0.61 * self.wavelength / self.NA
        res_image_plane_um = res_object_plane * self.magnification * 1e6
        self.psf_sigma_px = ((res_image_plane_um / self.pixel_size_um) / 2.355) * 1.3
        
        # Camera Parameters
        self.em_gain = 205.92    
        self.sensitivity = 4.88  
        
        # [수정] 오타 수정 및 누락된 파라미터 추가
        self.bias_offset = 457.80   # bias_offeset -> bias_offset
        self.cic_lambda = 0.0418    # Missing parameter added
        self.readout_sigma = 19.05  # Missing parameter added

# --- 2. NoiseFlow Loader ---
def load_noiseflow_model(ckpt_path, device="cuda"):
    hps = types.SimpleNamespace()
    hps.arch = "unc|unc|unc|unc|gain|unc|unc|unc|unc"
    hps.flow_permutation = 1 
    hps.lu_decomp = True
    hps.denoiser = "dncnn"
    hps.lmbda = 262144
    hps.x_shape = (1, 2, 64, 64)
    hps.device = device if torch.cuda.is_available() else "cpu"

    param_inits = init_params()
    model = Noise2NoiseFlow(
        hps.x_shape[1:],
        arch=hps.arch,
        flow_permutation=hps.flow_permutation,
        param_inits=param_inits,
        lu_decomp=hps.lu_decomp,
        denoiser_model=hps.denoiser,
        dncnn_num_layers=9,
        lmbda=hps.lmbda,
        device=hps.device,
    )

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=hps.device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(hps.device)
    model.eval()
    return model, hps

# --- 3. Noise Samplers ---
@torch.no_grad()
def sample_noise_from_flow(model, hps, bg_mean_frame, n_samples=1):
    denom = float(GLOBAL_VMAX - GLOBAL_VMIN)
    bg_norm = (bg_mean_frame - GLOBAL_VMIN) / denom
    bg_norm = np.clip(bg_norm, 0.0, 1.0).astype(np.float32)
    
    bg_t = torch.from_numpy(bg_norm).unsqueeze(0)
    bg_t = _ensure_channels(bg_t, C=hps.x_shape[1])
    bg_t = bg_t.unsqueeze(0).to(hps.device)
    
    noises = []
    for _ in range(n_samples):
        eps = model.noise_flow.sample(clean=bg_t)
        eps_np = eps[:, 0, :, :].cpu().numpy().squeeze()
        noise_adu = eps_np * denom
        full_bg = bg_mean_frame + noise_adu
        noises.append(full_bg)
    return np.stack(noises, axis=0)

def sample_noise_from_basden(config, shape, n_samples=1):
    H, W = shape
    noises = []
    eff_gain = config.em_gain / config.sensitivity
    
    for _ in range(n_samples):
        n_in = np.random.poisson(config.cic_lambda, size=(H, W))
        n_out = np.zeros_like(n_in, dtype=np.float32)
        mask = n_in > 0
        if np.any(mask):
            n_out[mask] = np.random.gamma(shape=n_in[mask], scale=eff_gain)
        
        # [수정] 오타 수정된 변수명 사용
        readout = np.random.normal(loc=config.bias_offset, scale=config.readout_sigma, size=(H, W))
        full_bg = n_out + readout
        noises.append(full_bg)
    return np.stack(noises, axis=0)

# --- Frequency Mixing ---
def mix_noise_in_frequency(flow_batch, basden_batch, cutoff_radius=3.0):
    B, H, W = flow_batch.shape
    mixed_batch = []
    
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
    mask = (x**2 + y**2) <= cutoff_radius**2
    
    for i in range(B):
        n_flow = flow_batch[i]
        n_basden = basden_batch[i]
        
        f_flow = np.fft.fft2(n_flow)
        f_basden = np.fft.fft2(n_basden)
        
        f_flow_s = np.fft.fftshift(f_flow)
        f_basden_s = np.fft.fftshift(f_basden)
        
        f_mixed_s = f_flow_s * mask + f_basden_s * (1 - mask)
        
        f_mixed = np.fft.ifftshift(f_mixed_s)
        n_mixed = np.fft.ifft2(f_mixed).real
        
        mixed_batch.append(n_mixed)
        
    return np.stack(mixed_batch, axis=0)


# --- 4. Signal Generator ---
def generate_atom_signal(config, t_exposure, grid_size=(64, 64), survival_prob=0.7):
    H, W = grid_size
    signal_canvas = np.zeros((H, W), dtype=np.float32)
    mu_photons = config.R_sc * t_exposure * config.eta_total
    
    spacing = 16.0  
    n_grid = 4
    center_y, center_x = H / 2.0, W / 2.0
    start_offset = (n_grid - 1) * spacing / 2.0
    start_y = center_y - start_offset
    start_x = center_x - start_offset
    
    atom_info_list = [] 
    thermal_jitter_px = 0.1

    for i in range(n_grid):
        for j in range(n_grid):
            loc_y = start_y + i * spacing
            loc_x = start_x + j * spacing
            
            is_occupied = 1 if np.random.random() < survival_prob else 0
            atom_info_list.append([loc_y, loc_x, is_occupied])
            
            if is_occupied:
                atom_efficiency = np.random.uniform(0.9, 1.1) 
                effective_mu = mu_photons * atom_efficiency

                n_photons = np.random.poisson(effective_mu)
                if n_photons > 0:
                    n_electrons = np.random.gamma(shape=n_photons, scale=config.em_gain)
                    
                    y_jit = loc_y + np.random.normal(0, thermal_jitter_px)
                    x_jit = loc_x + np.random.normal(0, thermal_jitter_px)
                    
                    margin = int(config.psf_sigma_px * 4)
                    y_min, y_max = max(0, int(y_jit)-margin), min(H, int(y_jit)+margin+1)
                    x_min, x_max = max(0, int(x_jit)-margin), min(W, int(x_jit)+margin+1)
                    
                    if y_min < y_max and x_min < x_max:
                        yy, xx = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
                        dist_sq = (xx - x_jit)**2 + (yy - y_jit)**2
                        psf = np.exp(-dist_sq / (2 * config.psf_sigma_px**2))
                        psf_sum = psf.sum()
                        if psf_sum > 0: psf /= psf_sum
                        signal_canvas[y_min:y_max, x_min:x_max] += psf * n_electrons

    return signal_canvas, np.array(atom_info_list)

# --- 5. Main Generation Pipeline ---
def generate_mixed_mock_data():
    cfg = SystemConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt_path = "experiments/archive/8-10-20-conseq.pth"
    model, hps = load_noiseflow_model(ckpt_path, device)
    
    bg_path = "./data_atom/data_atom_8_10_20_conseq/background.tif"
    bg_arr = imread(bg_path).astype(np.float32)
    bg_mean = bg_arr.mean(axis=0) if bg_arr.ndim == 3 else bg_arr

    # [디버깅] 신호 강도 확인용
    print(f"Signal Rate check: R={cfg.R_sc:.1e}, Eta={cfg.eta_total:.4f}")
    
    times = [2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3] 
    batch_size = 20000
    mix_cutoff = 5.0 
    
    save_dir = "./mock_dataset_hybrid_output"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating Hybrid Mock Data... Saving to {save_dir}")
    
    fig, axes = plt.subplots(1, len(times), figsize=(3*len(times), 4))
    if len(times) == 1: axes = [axes]

    for idx, t in enumerate(times):
        print(f"Processing Time: {t*1e3:.1f} ms...")
        
        batch_images = []   
        batch_labels = []   
        fixed_positions = None 

        for b in range(batch_size):
            signal_e, atom_info = generate_atom_signal(cfg, t, survival_prob=0.7)
            signal_adu = signal_e / cfg.sensitivity
            
            # [디버깅] 첫 배치에서 신호 레벨 출력
            if b == 0 and idx == 0:
                print(f"  -> Max Signal ADU: {signal_adu.max():.1f} (Noise Sigma ~{cfg.readout_sigma})")
            
            bg_flow = sample_noise_from_flow(model, hps, bg_mean, n_samples=2)
            bg_basden = sample_noise_from_basden(cfg, shape=(64,64), n_samples=2)
            
            bg_mixed = mix_noise_in_frequency(bg_flow, bg_basden, cutoff_radius=mix_cutoff)
            
            img1 = np.clip(signal_adu + bg_mixed[0], 0, 65535).astype(np.uint16)
            img2 = np.clip(signal_adu + bg_mixed[1], 0, 65535).astype(np.uint16)
            
            batch_images.append(np.stack([img1, img2], axis=0))
            batch_labels.append(atom_info[:, 2].astype(np.uint8))
            
            if b == 0:
                fixed_positions = atom_info[:, :2]

        final_imgs = np.stack(batch_images, axis=0) 
        final_lbls = np.stack(batch_labels, axis=0)
        
        tiff.imwrite(os.path.join(save_dir, f"images_{t*1e3:.0f}ms.tif"), final_imgs)
        np.save(os.path.join(save_dir, f"labels_{t*1e3:.0f}ms.npy"), final_lbls)
        np.save(os.path.join(save_dir, f"positions_{t*1e3:.0f}ms.npy"), fixed_positions)
        
        # Preview (Contrast 조정)
        # vmin을 Noise Bias(약 458) 근처로, vmax를 신호에 맞춰 조정
        ax = axes[idx]
        # 신호가 약할 수 있으므로 vmax를 낮게 설정하여 원자가 잘 보이게 함
        ax.imshow(final_imgs[0, 0], cmap='gray', vmin=450, vmax=700)
        ax.set_title(f"{t*1e3:.1f}ms")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "preview_hybrid.png"))
    plt.close()
    print("Done.")


def generate_bg_only(n_frames=100):
    # 1. 설정 및 모델 로드 (경로는 기존과 동일하게)
    cfg = SystemConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, hps = load_noiseflow_model("experiments/archive/8-10-20-conseq.pth", device)
    
    # 2. 배경 템플릿 로드
    bg_arr = imread("./data_atom/data_atom_8_10_20_conseq/background.tif")
    bg_mean = bg_arr.mean(axis=0) if bg_arr.ndim == 3 else bg_arr

    # 3. 노이즈 생성 및 합성 (Atom Signal 부분만 뺌)
    print(f"Generating {n_frames} Background Frames...")
    bg_flow = sample_noise_from_flow(model, hps, bg_mean, n_samples=n_frames)
    bg_basden = sample_noise_from_basden(cfg, shape=(64, 64), n_samples=n_frames)
    bg_mixed = mix_noise_in_frequency(bg_flow, bg_basden, cutoff_radius=5.0)

    # 4. 저장 (uint16 변환 필수)
    save_path = "./mock_dataset_hybrid_output/pure_bg_stack.tif"
    tiff.imwrite(save_path, np.clip(bg_mixed, 0, 65535).astype(np.uint16))
    print(f"Saved: {save_path}")
    

if __name__ == "__main__":
    generate_mixed_mock_data()
    generate_bg_only(n_frames=100)