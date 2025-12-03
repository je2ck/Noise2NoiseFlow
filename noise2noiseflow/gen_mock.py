import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import torch
from tifffile import imread
import os
import sys
import types
from scipy.stats import norm, poisson

# --- [Path Setup] ---
sys.path.append("../") 
from model.noise2noise_flow import Noise2NoiseFlow
from train_atom import init_params, _ensure_channels, GLOBAL_VMIN, GLOBAL_VMAX

# CUDA Hack
if not torch.cuda.is_available():
    def _fake_cuda(self, device=None, non_blocking=False): return self
    torch.Tensor.cuda = _fake_cuda
    torch.nn.Module.cuda = _fake_cuda

# --- 1. System Configuration ---
class SystemConfig:
    def __init__(self):
        self.R_sc = 0.8e5       # Calibrated Rate
        self.NA = 0.5
        self.wavelength = 556e-9
        self.pixel_size_um = 13.0
        self.magnification = 20.0
        self.T_optics = 0.8
        self.QE = 0.9
        
        self.eta_geo = (1 - np.sqrt(1 - self.NA**2)) / 2
        self.eta_total = self.eta_geo * self.T_optics * self.QE
        
        res_object_plane = 0.61 * self.wavelength / self.NA
        res_image_plane_um = res_object_plane * self.magnification * 1e6
        self.psf_sigma_px = ((res_image_plane_um / self.pixel_size_um) / 2.355) * 1.3
        
        self.em_gain = 300.0    
        self.sensitivity = 7.1  
        self.bias_offset = 457.84

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

# --- 3. Noise Sampler ---
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

# --- 4. Signal Generator (Returns occupancy mask too) ---
def generate_atom_signal(config, t_exposure, grid_size=(64, 64), survival_prob=0.7):
    H, W = grid_size
    signal_canvas = np.zeros((H, W), dtype=np.float32)
    
    mu_photons = config.R_sc * t_exposure * config.eta_total
    
    # 격자 생성
    spacing = 16.0  
    n_grid = 4
    center_y, center_x = H / 2.0, W / 2.0
    start_offset = (n_grid - 1) * spacing / 2.0
    start_y = center_y - start_offset
    start_x = center_x - start_offset
    
    atom_info_list = [] # [y, x, state]
    thermal_jitter_px = 0.1

    for i in range(n_grid):
        for j in range(n_grid):
            loc_y = start_y + i * spacing
            loc_x = start_x + j * spacing
            
            # Occupancy Check
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
def generate_flow_mock_data():
    cfg = SystemConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Load
    ckpt_path = "experiments/archive/8-10-20-conseq.pth"
    model, hps = load_noiseflow_model(ckpt_path, device)
    
    # Background Load
    bg_path = "./data_atom/data_atom_8_10_20_conseq/background.tif"
    bg_arr = imread(bg_path).astype(np.float32)
    bg_mean = bg_arr.mean(axis=0) if bg_arr.ndim == 3 else bg_arr

    # Settings
    times = [2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3] 
    batch_size = 100 # Time당 생성할 이미지 수 (배치 크기)
    
    save_dir = "./mock_dataset_output"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating Mock Data (Batch={batch_size}, Double Shot)...")
    
    # Preview용 Figure 생성
    fig, axes = plt.subplots(1, len(times), figsize=(3*len(times), 4))
    if len(times) == 1: axes = [axes]

    for idx, t in enumerate(times):
        print(f"Processing Time: {t*1e3:.1f} ms...")
        
        batch_images = []   # (Batch, 2, H, W)
        batch_labels = []   # (Batch, 16) -> Occupancy (0/1)
        fixed_positions = None # (16, 2) -> y, x 좌표 (모든 배치가 공유)

        for b in range(batch_size):
            # 1. Atom Geometry 생성 (점유 상태 결정)
            signal_e, atom_info = generate_atom_signal(cfg, t, survival_prob=0.7)
            # atom_info: [N, 3] -> col 0:y, col 1:x, col 2:state
            
            signal_adu = signal_e / cfg.sensitivity
            
            # 2. Double Shot Noise 생성 (같은 신호 + 다른 노이즈 2장)
            bg_frames = sample_noise_from_flow(model, hps, bg_mean, n_samples=2)
            
            # 3. Combine
            img1 = np.clip(signal_adu + bg_frames[0], 0, 65535).astype(np.uint16)
            img2 = np.clip(signal_adu + bg_frames[1], 0, 65535).astype(np.uint16)
            
            # Stack: (2, H, W)
            double_shot = np.stack([img1, img2], axis=0)
            batch_images.append(double_shot)
            
            # Labels: (N_sites,)
            occupancy = atom_info[:, 2].astype(np.uint8)
            batch_labels.append(occupancy)
            
            # Positions: (N_sites, 2) - 첫 번째 배치에서만 저장하면 됨 (격자는 고정이므로)
            if b == 0:
                fixed_positions = atom_info[:, :2] # y, x

        # --- Batch 저장 ---
        # Images: (Batch, 2, H, W)
        final_imgs_arr = np.stack(batch_images, axis=0)
        # Labels: (Batch, 16)
        final_labels_arr = np.stack(batch_labels, axis=0)
        
        # Save TIF (Image Stack)
        tiff.imwrite(os.path.join(save_dir, f"images_{t*1e3:.0f}ms.tif"), final_imgs_arr)
        
        # Save Labels (NPY)
        np.save(os.path.join(save_dir, f"labels_{t*1e3:.0f}ms.npy"), final_labels_arr)
        
        # Save Positions (NPY) - Time마다 하나씩 (사실 다 같지만 편의상)
        np.save(os.path.join(save_dir, f"positions_{t*1e3:.0f}ms.npy"), fixed_positions)
        
        # --- Preview (첫 번째 배치의 첫 번째 샷만) ---
        preview_img = final_imgs_arr[0, 0, :, :] # (H, W)
        preview_occ = final_labels_arr[0]        # (16,)
        preview_pos = fixed_positions            # (16, 2)
        
        ax = axes[idx]
        ax.imshow(preview_img, cmap='gray', vmin=450, vmax=700)
        
        # Occupied (Red O) / Empty (Blue X)
        # for k in range(len(preview_pos)):
        #     y, x = preview_pos[k]
        #     is_occ = preview_occ[k]
        #     if is_occ:
        #         ax.scatter(x, y, c='red', s=20, marker='o', facecolors='none', edgecolors='r')
        #     else:
        #         ax.scatter(x, y, c='blue', s=15, marker='x', alpha=0.5)
                
        ax.set_title(f"{t*1e3:.1f} ms")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "preview_summary.png"))
    plt.show()
    print(f"Done. All batch files saved in '{save_dir}'")

if __name__ == "__main__":
    generate_flow_mock_data()