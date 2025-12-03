import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile import imread
import os
from scipy.stats import norm, poisson
from scipy import fftpack


import sys
sys.path.append("../")

from model.noise2noise_flow import Noise2NoiseFlow
from train_atom import init_params, _ensure_channels, GLOBAL_VMIN, GLOBAL_VMAX
import types


# CUDA 없는 환경에서 .cuda() 호출을 CPU no-op 으로 바꾸는 꼼수
if not torch.cuda.is_available():
    print("[hack] CUDA is not available -> patching .cuda() to be a no-op on CPU")

    # Tensor.cuda(...) -> 그냥 self 반환 (CPU에 그대로)
    def _fake_tensor_cuda(self, device=None, non_blocking=False):
        return self  # 이미 CPU tensor

    # Module.cuda(...) -> 그냥 self 반환
    def _fake_module_cuda(self, device=None):
        return self  # 파라미터들 그대로 CPU

    torch.Tensor.cuda = _fake_tensor_cuda
    torch.nn.Module.cuda = _fake_module_cuda 

def get_radial_profile(image):
    """이미지 한 장의 Radial PSD 프로파일을 반환"""
    # DC 성분 제거 (중요)
    image = image - image.mean()
    
    f = fftpack.fft2(image)
    fshift = fftpack.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-12)
    
    h, w = image.shape
    center = (h//2, w//2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # 빈도수 계산 (bincount)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    
    # 0으로 나누기 방지
    radialprofile = tbin / (nr + 1e-12)
    
    # 중심(DC)부터 가장자리까지
    return radialprofile[:h//2]

def plot_average_psd(image_stack, label, color, ax):
    """
    image_stack: (N, H, W) 이미지 묶음
    여러 장의 PSD를 평균내서 plot (Mean + Std shading)
    """
    profiles = []
    for i in range(len(image_stack)):
        prof = get_radial_profile(image_stack[i])
        profiles.append(prof)
    
    profiles = np.array(profiles) # (N, Radius)
    
    # 평균과 표준편차 계산
    mean_prof = profiles.mean(axis=0)
    std_prof = profiles.std(axis=0)
    
    x_axis = np.arange(len(mean_prof))
    
    # 평균선 그리기
    ax.plot(x_axis, mean_prof, color=color, label=label, linewidth=2)
    # 표준편차 범위(불확실성) 색칠하기
    ax.fill_between(x_axis, mean_prof - std_prof, mean_prof + std_prof, color=color, alpha=0.2)
    
    
def plot_radial_psd(image, label, color, ax):
    """
    이미지의 방사형 평균 PSD(Power Spectral Density)를 계산하고 ax에 그립니다.
    image: 2D numpy array
    """
    f = fftpack.fft2(image)
    fshift = fftpack.fftshift(f)
    # Power Spectrum (log scale dB)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-12)
    
    h, w = image.shape
    center = (h//2, w//2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # Radial Profile 계산 (거리에 따른 평균 파워)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / (nr + 1e-12)
    
    # DC 성분(0번) 제외하고 그리기
    ax.plot(radialprofile[1:h//2], label=label, color=color, alpha=0.8, linewidth=2)
    
    
def sample_basden_emccd_frames(mean_frame_adu, n_frames, em_gain, sigma_read, bias_offset=457.84):
    """
    Basden Model (Compound Poisson-Gamma) + Readout Noise를 이용해 샘플 생성
    
    1. Input Photon: Poisson(lambda)
    2. EM Amplification: Gamma(k, scale=gain) (Total sum of k exponentials)
    3. Readout Noise: Gaussian(0, sigma)
    
    mean_frame_adu: (H,W) 평균 밝기 (이를 통해 입력 광자 수 lambda를 역산)
    """
    H, W = mean_frame_adu.shape
    
    # 1. 입력 광자 수(Lambda) 역산 (ADU -> Electron)
    # 가정: Mean Signal ≈ (Lambda * Gain) + Bias
    # Lambda ≈ (Mean - Bias) / Gain
    # 음수가 나오지 않도록 clip
    lam_map = (mean_frame_adu - bias_offset) / em_gain
    lam_map = np.maximum(lam_map, 1e-9) # 0 방지
    
    # (N, H, W) 크기로 확장
    lam_expanded = np.repeat(lam_map[None, ...], n_frames, axis=0)
    
    # 2. Poisson Sampling (입력 전자 수)
    n_in = np.random.poisson(lam_expanded)
    
    # 3. EM Gain Amplification (Gamma Sampling)
    # n_in개의 전자가 각각 Exp(gain)을 따르므로, 합은 Gamma(n_in, gain)
    # n_in = 0인 경우 Gamma는 0이어야 함.
    # numpy.random.gamma는 shape=0일 때 0.0을 반환하거나 에러가 날 수 있으므로 마스킹 처리
    n_out = np.zeros_like(n_in, dtype=np.float32)
    
    mask = n_in > 0
    if np.any(mask):
        # shape: 입력 전자 수, scale: 증폭 이득
        n_out[mask] = np.random.gamma(shape=n_in[mask], scale=em_gain)
        
    # 4. Readout Noise 더하기 & Bias 복구
    readout = np.random.normal(loc=bias_offset, scale=sigma_read, size=(n_frames, H, W))
    
    final_frame = n_out + readout
    
    return final_frame.astype(np.float32)

def gaussian_fit_pmf(bins, samples):
    """
    bins: (K+1,)
    samples: 1D array noise samples
    
    Return: gaussian_pmf(K,)
    """
    mu = samples.mean()
    sigma = samples.std() + 1e-12

    # PDF at bin centers
    centers = 0.5 * (bins[:-1] + bins[1:])
    pdf = norm.pdf(centers, loc=mu, scale=sigma)

    # Convert pdf → pmf
    widths = np.diff(bins)
    pmf = pdf * widths
    pmf /= pmf.sum()

    return pmf, mu, sigma

def poisson_gaussian_fit_pmf(bins, samples, max_poisson_mult=5):
    """
    Fit Poisson-Gaussian model:
        X = Poisson(lambda) + N(0, sigma^2)

    bins: histogram bin boundaries
    samples: 1D noise samples (mean≈0 after subtract_mean)

    Return:
        pmf (K,)
        lambda_hat
        sigma_hat
    """
    # Step 1: parameter estimation
    mean_x = samples.mean()
    var_x = samples.var()

    lam = mean_x
    sigma2 = max(var_x - lam, 1e-12)
    sigma = np.sqrt(sigma2)

    centers = 0.5 * (bins[:-1] + bins[1:])
    widths  = np.diff(bins)
    K = len(centers)

    # Step 2: Build Poisson-Gaussian pdf on centers
    # Range of Poisson values to consider
    max_k = int(lam + max_poisson_mult * np.sqrt(lam + 1e-6)) + 10
    ks = np.arange(0, max_k)

    # Poisson prob
    pois_p = poisson.pmf(ks, lam)

    # For each k, gaussian centered at k
    pdf = np.zeros_like(centers)
    for i, k in enumerate(ks):
        pdf += pois_p[i] * norm.pdf(centers, loc=k, scale=sigma)

    # Step 3: PDF → PMF
    pmf = pdf * widths
    pmf /= pmf.sum()

    return pmf, lam, sigma

def compute_pmf_from_tiff_stack(
    tiff_path,
    num_bins=200,
    subtract_mean=True,
    roi=None,
):
    imgs = tiff.imread(tiff_path).astype(np.float32)  # (T,H,W)

    if imgs.ndim != 3:
        raise ValueError("TIFF must be (T,H,W)")

    # ROI
    if roi is not None:
        y0, y1, x0, x1 = roi
        imgs = imgs[:, y0:y1, x0:x1]

    # ---- 핵심: 프레임마다 별도로 평균 제거 ----
    if subtract_mean:
        frame_means = imgs.mean(axis=(1,2), keepdims=True)  # (T,1,1)
        imgs = imgs - frame_means

    samples = imgs.reshape(-1)

    vmin, vmax = samples.min(), samples.max()
    bins = np.linspace(vmin, vmax, num_bins+1)

    hist, _ = np.histogram(samples, bins=bins, density=False)
    pmf = hist / hist.sum()

    return bins, pmf

def visualize_pmf_save(bins, pmf, out_path, title="PMF of Noise Distribution"):
    centers = 0.5 * (bins[:-1] + bins[1:])  # bin 중심

    plt.figure(figsize=(10, 5))

    plt.plot(centers, pmf, '-o', markersize=2, label='PMF (line)')
    plt.bar(centers, pmf, width=np.diff(bins), alpha=0.3, label='PMF (bar)')
    plt.step(bins[:-1], pmf, where='post', color='red', alpha=0.5, label='PMF (step)')

    plt.xlabel("Noise / Pixel Value")
    plt.ylabel("Probability (pmf)")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    # 화면에 띄우지 않고 저장만
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[saved PMF figure] -> {out_path}")


# ----------------------
# 1) 하이퍼파라미터 & 모델 로드 (flow까지 포함)
# ----------------------
def build_hps(device="cuda"):
    hps = types.SimpleNamespace()
    hps.arch = "unc|unc|unc|unc|gain|unc|unc|unc|unc"
    hps.flow_permutation = 1 
    hps.lu_decomp = True
    hps.denoiser = "dncnn"
    hps.lmbda = 262144

    C, H, W = 2, 64, 64
    hps.x_shape = (1, C, H, W)
    hps.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    return hps


def load_trained_model_for_flow(ckpt_path: str, device="cuda"):
    """
    기존 load_trained_model은 denoiser.*만 로드하니까,
    flow까지 포함해서 전체 state_dict를 읽는 버전.
    """
    hps = build_hps(device=device)
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

    ckpt = torch.load(ckpt_path, map_location=hps.device)
    state = ckpt["state_dict"]
    model.load_state_dict(state, strict=True)

    model.to(hps.device)
    model.eval()
    return model, hps


# ----------------------
# 2) background 스택에서 bg mean (클린 배경) 만들기
# ----------------------
def build_bg_mean_tensor(bg_tif_path: str, hps):
    """
    bg_tif_path: (N,H,W) 또는 (H,W) background raw TIF
    return: bg_mean_t (1,C,H,W), [0,1] normalized, hps.device
    """
    arr = imread(bg_tif_path).astype(np.float32)  # (N,H,W) or (H,W)

    if arr.ndim == 2:
        arr = arr[None, ...]  # (1,H,W)

    # (N,H,W) → mean over frames → (H,W)
    bg_mean_raw = arr.mean(axis=0)

    # 학습과 동일 정규화: (x - VMIN) / (VMAX - VMIN)
    denom = float(GLOBAL_VMAX - GLOBAL_VMIN)
    bg_mean_norm = (bg_mean_raw - GLOBAL_VMIN) / denom
    bg_mean_norm = np.clip(bg_mean_norm, 0.0, 1.0).astype(np.float32)  # (H,W)

    # (H,W) → (1,H,W) → _ensure_channels → (C,H,W) → (1,C,H,W)
    bg_t = torch.from_numpy(bg_mean_norm).unsqueeze(0)           # (1,H,W)
    bg_t = _ensure_channels(bg_t, C=hps.x_shape[1])              # (C,H,W)
    bg_t = bg_t.unsqueeze(0).to(hps.device)                      # (1,C,H,W)
    return bg_t, denom


# ----------------------
# 3) noise model에서 noise 샘플 뽑고 noisy bg 만들기
# ----------------------
@torch.no_grad()
def sample_noisy_bg_from_flow(model, bg_t, n_samples=32):
    """
    model : Noise2NoiseFlow (model.flow가 NoiseFlow 인스턴스라고 가정)
    bg_t  : (1,C,H,W) clean background (normalized [0,1])
    return:
        noisy_norm: (n_samples, H, W)  [0,1]
        noise_norm: (n_samples, H, W)  [0,1] (bg 기준 noise)
    """
    device = bg_t.device
    B, C, H, W = bg_t.shape
    assert B == 1

    noisy_list = []
    noise_list = []

    for _ in range(n_samples):
        # NoiseFlow의 sample()을 그대로 사용
        # clean=bg_t 를 넘겨줘야 prior가 올바른 shape와 스케일로 z를 만든다.
        eps = model.noise_flow.sample(clean=bg_t)        # (1,C,H,W)  = noise sample

        eps_ch0 = eps[:, 0, :, :]                  # (1,H,W)
        bg_ch0  = bg_t[:, 0, :, :]                 # (1,H,W)

        y = bg_ch0 + eps_ch0                       # (1,H,W)  = noisy image

        noise_list.append(eps_ch0.squeeze(0).cpu().numpy())   # (H,W)
        noisy_list.append(y.squeeze(0).cpu().numpy())         # (H,W)

    noisy_norm = np.stack(noisy_list, axis=0)      # (N,H,W)
    noise_norm = np.stack(noise_list, axis=0)      # (N,H,W)

    return noisy_norm, noise_norm


# ----------------------
# 4) noisy/ noise 로부터 pmf / pdf 계산
# ----------------------
def compute_pmf_pdf_from_images(images, num_bins=200, subtract_mean=True):
    """
    images: (N,H,W) or (H,W), float32
    return:
        bins: (K+1,), pmf: (K,), pdf: (K,)
    """
    if images.ndim == 2:
        images = images[None, ...]
    N, H, W = images.shape

    imgs = images.astype(np.float32)

    if subtract_mean:
        mean_val = imgs.mean()
        imgs = imgs - mean_val

    samples = imgs.reshape(-1)  # (N*H*W,)

    vmin, vmax = samples.min(), samples.max()
    bins = np.linspace(vmin, vmax, num_bins + 1)

    # pmf: count 정규화
    hist, _ = np.histogram(samples, bins=bins, density=False)
    pmf = hist / hist.sum()

    # pdf 근사: density=True
    hist_dens, _ = np.histogram(samples, bins=bins, density=True)
    pdf = hist_dens

    return bins, pmf, pdf

def save_background_sample(bg_tif_path, out_dir="./noiseflow_viz"):
    """
    background TIFF에서 첫 이미지를 꺼내서 저장.
    PNG(보기용) + RAW TIF(정확한 데이터) 모두 저장.
    """
    os.makedirs(out_dir, exist_ok=True)

    arr = imread(bg_tif_path).astype(np.float32)  # (N,H,W) or (H,W)

    # (H,W) 맞추기
    if arr.ndim == 3:
        bg_raw = arr[0]      # 첫 이미지
    elif arr.ndim == 2:
        bg_raw = arr
    else:
        raise ValueError("Unsupported TIFF shape.")

    # PNG (normalize해서 보기 좋게)
    bg_norm = (bg_raw - bg_raw.min()) / (bg_raw.max() - bg_raw.min() + 1e-8)

    plt.figure()
    plt.imshow(bg_norm, cmap="gray")
    plt.colorbar()
    plt.title("Raw background frame (normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "background_frame_norm.png"), dpi=200)
    plt.close()

    # RAW TIF 그대로 저장
    # tiff.imwrite(
    #     os.path.join(out_dir, "background_frame_raw.tif"),
    #     bg_raw.astype(np.uint16) if bg_raw.max() > 255 else bg_raw.astype(np.uint8)
    # )

    print("[saved original background image]")
    
def visualize_pmf_pdf_save(bins, pmf, pdf, out_path, title="Noise PMF/PDF"):
    centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(10, 5))
    plt.plot(centers, pmf, '-o', markersize=2, label='PMF (discrete)', alpha=0.8)
    plt.plot(centers, pdf, '-', label='PDF (density=True)', alpha=0.8)
    plt.xlabel("Noise / Pixel Value")
    plt.ylabel("Probability / Density")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
def save_example_images(
    bg_t,
    noise_norm,
    noisy_norm,
    denom,
    out_dir="./noiseflow_viz",
    max_examples=4,
):
    """
    bg_t      : (1,C,H,W), [0,1]
    noise_norm: (N,H,W),   [0,1] (bg 기준 noise)
    noisy_norm: (N,H,W),   [0,1] (bg + noise)
    denom     : (GLOBAL_VMAX - GLOBAL_VMIN)
    out_dir   : 저장 폴더
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) bg_mean (정규화 & raw 둘 다) ----
    bg_norm = bg_t[:, 0, :, :].squeeze(0).detach().cpu().numpy()  # (H,W), [0,1]
    bg_raw  = bg_norm * denom + GLOBAL_VMIN                       # (H,W), raw 스케일

    # PNG (보기용)
    plt.figure()
    plt.imshow(bg_norm, cmap="gray")
    plt.colorbar()
    plt.title("Background mean (normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bg_mean_norm.png"), dpi=200)
    plt.close()

    # # TIF (raw 스케일)
    # tiff.imwrite(
    #     os.path.join(out_dir, "bg_mean_raw.tif"),
    #     bg_raw.astype(np.uint16),
    # )

    # ---- 2) noise / noisy 예시 이미지 몇 장 저장 ----
    N, H, W = noise_norm.shape
    n_save = min(max_examples, N)

    for i in range(n_save):
        noise_i = noise_norm[i]      # (H,W), [0,1] 기준 noise
        noisy_i = noisy_norm[i]      # (H,W), [0,1] noisy image

        # 보기용 PNG (noise는 contrast 때문에 mean 0 근처일 수 있어서 조금 조심)
        plt.figure(figsize=(8, 3))

        plt.subplot(1, 2, 1)
        plt.imshow(noise_i, cmap="gray")
        plt.title(f"Noise sample {i}")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(noisy_i, cmap="gray")
        plt.title(f"Noisy (bg + noise) {i}")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"noise_and_noisy_{i}.png"), dpi=200)
        plt.close()

        # 원하면 raw 스케일로 TIF 저장도 가능
        noise_raw = noise_i * denom              # noise 자체는 offset 없음
        noisy_raw = noisy_i * denom + bg_raw     # bg_raw + noise_raw

        # tiff.imwrite(
        #     os.path.join(out_dir, f"noise_raw_{i}.tif"),
        #     noise_raw.astype(np.float32),  # noise는 float로 두는 게 나을 수 있음
        # )
        # tiff.imwrite(
        #     os.path.join(out_dir, f"noisy_raw_{i}.tif"),
        #     np.clip(noisy_raw, 0, 65535).astype(np.uint16),
        # )

def kl_divergence_pmf(p_bins, p_pmf, q_bins, q_pmf, eps=1e-12):
    """
    p_bins, q_bins: bin boundaries (same length ideally)
    p_pmf, q_pmf  : discrete pmf arrays of shape (K,)
    
    Returns: scalar KL(P || Q)
    """
    # --- 1) bin 일치 여부 체크 ---
    if not np.allclose(p_bins, q_bins):
        raise ValueError("P and Q must use identical bin boundaries for PMF KL divergence.")

    # --- 2) Normalize just in case ---
    p = p_pmf / (p_pmf.sum() + eps)
    q = q_pmf / (q_pmf.sum() + eps)

    # --- 3) KL(P||Q) ---
    mask = p > 0
    kl = np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps)))
    return kl


def extract_noise_samples_from_background(bg_tif_path, roi=None):
    imgs = tiff.imread(bg_tif_path).astype(np.float32)  # (T,H,W)

    if imgs.ndim != 3:
        raise ValueError("TIFF must be (T,H,W)")

    if roi is not None:
        y0, y1, x0, x1 = roi
        imgs = imgs[:, y0:y1, x0:x1]

    # frame-wise mean subtraction (중요)
    frame_means = imgs.mean(axis=(1,2), keepdims=True)
    imgs = imgs - frame_means

    samples_raw = imgs.reshape(-1)  # raw noise (mean≈0)

    # 학습 때와 같은 0~1 정규화 스케일로 맞추기
    denom = (GLOBAL_VMAX - GLOBAL_VMIN)
    bg_norm_samples = (samples_raw - GLOBAL_VMIN) / denom

    # NoiseFlow쪽과 공정하게 비교하려고 전체 평균 0으로 맞춤
    bg_norm_samples = bg_norm_samples - bg_norm_samples.mean()

    return bg_norm_samples

def pmf_with_common_bins(all_samples, samples_p, samples_q, num_bins=200):
    vmin, vmax = all_samples.min(), all_samples.max()
    bins = np.linspace(vmin, vmax, 200)

    bins = np.linspace(vmin, vmax, num_bins+1)

    hist_p, _ = np.histogram(samples_p, bins=bins, density=False)
    hist_q, _ = np.histogram(samples_q, bins=bins, density=False)

    pmf_p = hist_p / hist_p.sum()
    pmf_q = hist_q / hist_q.sum()

    return bins, pmf_p, pmf_q


def compare_gaussian_noise_model():
    ckpt_path = "experiments/weights/best_model_real.pth"
    bg_stack_path = "./data_atom/background.tif"  # (N,H,W) 또는 (H,W)

    # ---------------------------------------------------
    # 0) RAW background noise PMF (optional visualization)
    # ---------------------------------------------------
    bins_p, pmf_raw = compute_pmf_from_tiff_stack(
        tiff_path=bg_stack_path,
        num_bins=200,
        subtract_mean=True,
        roi=None,
    )
    visualize_pmf_save(
        bins_p,
        pmf_raw,
        out_path="./noiseflow_viz/pmf_raw_background_stack.png",
        title="PMF of Raw Background Stack"
    )

    # ---------------------------------------------------
    # 1) NoiseFlow 전체 모델 로드
    # ---------------------------------------------------
    model, hps = load_trained_model_for_flow(ckpt_path, device="cuda")

    # ---------------------------------------------------
    # 2) Background mean (clean bg image) → bg_t
    # ---------------------------------------------------
    bg_t, denom = build_bg_mean_tensor(bg_stack_path, hps)

    # ---------------------------------------------------
    # 3) NoiseFlow noise 샘플 생성
    # ---------------------------------------------------
    noisy_norm, noise_norm = sample_noisy_bg_from_flow(
        model,
        bg_t,
        n_samples=1024,
    )

    # ---------------------------------------------------
    # 4) Background stack에서 noise samples 추출 (1D)
    # ---------------------------------------------------
    bg_norm_samples = extract_noise_samples_from_background(bg_stack_path)

    # ---------------------------------------------------
    # 5) NoiseFlow noise sample을 1D로 변환
    # ---------------------------------------------------
    # noise_norm: (N,H,W) normalized residual noise
    # noise_norm: (N,H,W), 이미 (x - VMIN)/(VMAX - VMIN) 스케일이라고 가정
    noise_norm_samples = noise_norm.reshape(-1).astype(np.float32)

    # 배경과 마찬가지로 전체 평균 0으로 맞추기
    noise_norm_samples = noise_norm_samples - noise_norm_samples.mean()

    # ---------------------------------------------------
    # 6) 공통 bins로 두 PMF 계산
    # ---------------------------------------------------
    all_samples = np.concatenate([bg_norm_samples, noise_norm_samples], axis=0)
    bins, pmf_bg, pmf_nf = pmf_with_common_bins(
        all_samples,
        bg_norm_samples,
        noise_norm_samples,
        num_bins=200
    )

    # ---------------------------------------------------
    # 7) KL(bg || noiseflow) 계산
    # ---------------------------------------------------
    KL = kl_divergence_pmf(bins, pmf_bg, bins, pmf_nf)
    print("KL(background || NoiseFlow) =", KL)

    # ---------------------------------------------------
    # 8) PMF/PDF 시각화 (NoiseFlow 쪽)
    # ---------------------------------------------------
    # NoiseFlow에 대해만 PDF 계산
    bins_nf, pmf_nf_single, pdf_nf_single = compute_pmf_pdf_from_images(
    noise_norm, num_bins=200, subtract_mean=True
    )
    visualize_pmf_pdf_save(
        bins_nf, pmf_nf_single, pdf_nf_single,
        out_path="./noiseflow_viz/pmf_pdf_noiseflow_local.png",
        title="NoiseFlow Noise (local bins)"
    )

    # ---------------------------------------------------
    # 9) 예시 이미지 저장
    # ---------------------------------------------------
    save_example_images(
        bg_t=bg_t,
        noise_norm=noise_norm,
        noisy_norm=noisy_norm,
        denom=denom,
        out_dir="./noiseflow_viz",
        max_examples=1,
    )

    save_background_sample(bg_stack_path, out_dir="./noiseflow_viz")

    # Gaussian fit PMF
    pmf_gauss, mu_g, sigma_g = gaussian_fit_pmf(bins, bg_norm_samples)
    KL_gauss = kl_divergence_pmf(bins, pmf_bg, bins, pmf_gauss)
    print("KL(background || Gaussian) =", KL_gauss)

    # 기존 NoiseFlow KL
    KL_nf = kl_divergence_pmf(bins, pmf_bg, bins, pmf_nf)
    print("KL(background || NoiseFlow) =", KL_nf)

def compare_poisson_gaussian_noise_model(
    n_kl_samples=30,      
    Nf=512,               
    num_bins=200,
    em_gain=42.66,        # [New] EM Gain 설정 (실험 조건에 맞게 변경: 300, 1000 등)
    bias_offset=457.84       # [New] Bias (Offset) 설정. 보통 mean의 최솟값 혹은 캘리브레이션 값
):
    ckpt_path = "experiments/archive/8-10-20-conseq.pth"
    bg_stack_path = "./data_atom/data_atom_8_10_20_conseq/background.tif"
    out_dir = "./noiseflow_viz"
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # 1) 실제 background raw 스택 로드
    # -----------------------------
    arr = imread(bg_stack_path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    T, H, W = arr.shape

    bg_samples_raw = arr.reshape(-1)
    mean_frame = arr.mean(axis=0) # (H,W)

    # Bias 자동 추정 (단순하게 데이터 최솟값 근처로 잡거나 0으로 가정)
    # 정확한 비교를 위해선 실험 세팅의 Bias 값을 넣는 게 좋습니다.
    # 여기선 데이터의 최소값보다 약간 작은 값을 Bias로 가정하거나 0으로 둡니다.
    if bias_offset == 0.0:
        bias_offset = np.percentile(bg_samples_raw, 1) # 하위 1%를 bias로 추정 (임시)

    # -----------------------------
    # 2) 파라미터 추정 (Gaussian / Poisson-Gaussian / Basden Readout)
    # -----------------------------
    mean_bg = bg_samples_raw.mean()
    var_bg  = bg_samples_raw.var()

    # (A) Gaussian
    sigma_g = np.sqrt(max(var_bg, 1e-12))

    # (B) Poisson-Gaussian
    lam_pg  = max(mean_bg, 1e-6)
    sigma2_pg = max(var_bg - lam_pg, 1e-12)
    sig_pg  = np.sqrt(sigma2_pg)
    
    # (C) [New] Basden Model Parameter
    # Basden 모델에서 Readout Noise(sigma_read)는 
    # 보통 EM Gain이 적용되지 않은 순수 회로 노이즈입니다.
    # 여기서는 데이터의 variance에서 Shot noise 성분을 뺀 나머지를 추정하거나
    # 카메라 스펙의 readout noise를 넣어야 합니다.
    # 약식으로: Var_total approx (2 * Mean * Gain) + Sigma_read^2 (High gain approx)
    # 하지만 여기선 그냥 작은 값 혹은 G/PG와 비슷한 수준으로 가정해봅니다.
    sigma_read_basden = 18.91 # 예시: 50 ADU (실험값에 맞춰 수정 필요)

    # -----------------------------
    # 3) NoiseFlow 로드
    # -----------------------------
    model, hps = load_trained_model_for_flow(ckpt_path, device="cuda")
    denom = float(GLOBAL_VMAX - GLOBAL_VMIN)

    # NoiseFlow 샘플링 함수 (기존과 동일)
    @torch.no_grad()
    def sample_nf_frames_from_mean(n_frames):
        frames = []
        clean_norm = (mean_frame - GLOBAL_VMIN) / denom
        clean_norm = np.clip(clean_norm, 0.0, 1.0).astype(np.float32)
        clean_t = torch.from_numpy(clean_norm).unsqueeze(0)
        clean_t = _ensure_channels(clean_t, C=hps.x_shape[1])
        clean_t = clean_t.unsqueeze(0).to(hps.device)

        for _ in range(n_frames):
            eps = model.noise_flow.sample(clean=clean_t)
            eps_raw = eps[:, 0].detach().cpu().numpy()[0] * denom
            frame_raw = mean_frame + eps_raw
            frames.append(frame_raw)
        return np.stack(frames, axis=0)

    # -----------------------------
    # 4) Bins 설정
    # -----------------------------
    vmin, vmax = bg_samples_raw.min(), bg_samples_raw.max()
    margin = 0.05 * (vmax - vmin + 1e-12)
    bins = np.linspace(vmin - margin, vmax + margin, num_bins + 1)

    def pmf_from(samples):
        hist, _ = np.histogram(samples, bins=bins, density=False)
        pmf = hist / (hist.sum() + 1e-12)
        return pmf

    pmf_bg = pmf_from(bg_samples_raw)

    # -----------------------------
    # 5) Main Loop: Sample & Calc KL
    # -----------------------------
    KL_gauss_list = []
    KL_pg_list    = []
    KL_nf_list    = []
    KL_basden_list = [] # [New]

    first_gauss_syn = None
    first_pg_syn    = None
    first_nf_syn    = None
    first_basden_syn = None # [New]

    print(f"Start KL comparison loop (Gain={em_gain}, Bias={bias_offset:.1f})...")

    for trial in range(n_kl_samples):
        # (a) NoiseFlow
        nf_syn = sample_nf_frames_from_mean(Nf)
        nf_samples = nf_syn.reshape(-1)

        # (b) Gaussian
        gauss_residual = np.random.normal(0.0, sigma_g, size=(Nf, H, W)).astype(np.float32)
        gauss_syn = mean_frame[None, ...] + gauss_residual
        gauss_samples = gauss_syn.reshape(-1)

        # (c) Poisson-Gaussian
        S = np.random.poisson(lam_pg, size=(Nf, H, W)).astype(np.float32)
        G = np.random.normal(0.0, sig_pg, size=(Nf, H, W)).astype(np.float32)
        pg_syn = mean_frame[None, ...] + (S - lam_pg) + G
        pg_samples = pg_syn.reshape(-1)
        
        # 피팅된 CIC Lambda 값 (0.0041)
        fitted_cic_lambda = 0.0041 
        
        # sample_basden_emccd_frames 함수는 mean_frame을 받아서 lambda를 계산하도록 되어 있으므로,
        # lambda 값을 직접 받는 새로운 함수를 쓰거나, 
        # mean_frame 자리에 (fitted_cic_lambda * em_gain + bias_offset) 값을 가진 가짜 프레임을 넣어주면 됩니다.
        
        # 방법 1: 가짜 mean_frame을 만들어서 넘겨주기 (함수 수정 없이 가능)
        synthetic_mean_val = (fitted_cic_lambda * em_gain) + bias_offset
        synthetic_mean_frame = np.full((H, W), synthetic_mean_val, dtype=np.float32)
        
        # (d) [New] Basden (EMCCD) Model
        # mean_frame을 기준으로 Lambda 맵을 만들어 샘플링
        basden_syn = sample_basden_emccd_frames(
            mean_frame_adu=synthetic_mean_frame, 
            n_frames=Nf, 
            em_gain=em_gain, 
            sigma_read=sigma_read_basden,
            bias_offset=bias_offset
        )
        basden_samples = basden_syn.reshape(-1)

        # 저장 (첫 trial)
        if trial == 0:
            first_gauss_syn = gauss_syn.copy()
            first_pg_syn    = pg_syn.copy()
            first_nf_syn    = nf_syn.copy()
            first_basden_syn = basden_syn.copy()

        # PMF & KL
        pmf_g  = pmf_from(gauss_samples)
        pmf_pg = pmf_from(pg_samples)
        pmf_nf = pmf_from(nf_samples)
        pmf_bd = pmf_from(basden_samples) # [New]

        kl_g  = kl_divergence_pmf(bins, pmf_bg, bins, pmf_g)
        kl_pg = kl_divergence_pmf(bins, pmf_bg, bins, pmf_pg)
        kl_nf = kl_divergence_pmf(bins, pmf_bg, bins, pmf_nf)
        kl_bd = kl_divergence_pmf(bins, pmf_bg, bins, pmf_bd) # [New]

        KL_gauss_list.append(kl_g)
        KL_pg_list.append(kl_pg)
        KL_nf_list.append(kl_nf)
        KL_basden_list.append(kl_bd)

        print(f"[trial {trial+1}] KL_G={kl_g:.4f}, KL_PG={kl_pg:.4f}, KL_Basden={kl_bd:.4f}, KL_NF={kl_nf:.4f}")

    # 결과 정리
    KL_gauss_arr = np.array(KL_gauss_list)
    KL_pg_arr    = np.array(KL_pg_list)
    KL_nf_arr    = np.array(KL_nf_list)
    KL_bd_arr    = np.array(KL_basden_list)

    print("\n=== Mean KL Divergences ===")
    print(f"Gaussian        : {KL_gauss_arr.mean():.6f}")
    print(f"Poisson-Gaussian: {KL_pg_arr.mean():.6f}")
    print(f"Basden (EMCCD)  : {KL_bd_arr.mean():.6f}")
    print(f"NoiseFlow       : {KL_nf_arr.mean():.6f}")

    # -----------------------------
    # 6) Boxplot 저장 (Basden 포함)
    # -----------------------------
    plt.figure(figsize=(8, 5))
    data = [KL_gauss_arr, KL_pg_arr, KL_bd_arr, KL_nf_arr]
    labels = ["Gaussian", "Poisson-Gauss", "Basden(EMCCD)", "NoiseFlow"]
    
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("KL Divergence (lower is better)")
    plt.title(f"Noise Model Comparison (Gain={em_gain})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "KL_boxplot_with_basden.png"), dpi=200)
    plt.close()

    # -----------------------------
    # 7) synthetic background 예시 이미지 저장 (첫 trial 기준)
    # -----------------------------
    def _save_raw_image(raw_img, out_path, title):
        img_norm = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        plt.figure()
        plt.imshow(img_norm, cmap="gray")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    # (a) Real background (한 프레임)
    real_bg = arr[0]
    _save_raw_image(real_bg,
                    os.path.join(out_dir, "bg_real_raw.png"),
                    "Real background (one frame, raw)")

    # (b) Gaussian model synthetic (첫 trial)
    _save_raw_image(first_gauss_syn[0],
                    os.path.join(out_dir, "bg_gaussian_model_raw.png"),
                    "Gaussian model synthetic background")

    # (c) Poisson-Gaussian model synthetic (첫 trial)
    _save_raw_image(first_pg_syn[0],
                    os.path.join(out_dir, "bg_poisson_gaussian_model_raw.png"),
                    "Poisson-Gaussian model synthetic background")

    # (d) NoiseFlow model synthetic (첫 trial)
    _save_raw_image(first_nf_syn[0],
                    os.path.join(out_dir, "bg_noiseflow_model_raw.png"),
                    "NoiseFlow model synthetic background")
    
    # (e) Basden model synthetic (첫 trial)
    _save_raw_image(first_basden_syn[0],
                    os.path.join(out_dir, "bg_basden_model_raw.png"),
                    "Basden model synthetic background")


    # -----------------------------
    # 8) PMF 비교 플롯 (첫 trial 기준)
    # -----------------------------
    # 첫 trial에서 사용했던 샘플을 다시 한 번 pmf_from에 넣어 사용
    pmf_gauss_first = pmf_from(first_gauss_syn.reshape(-1))
    pmf_pg_first    = pmf_from(first_pg_syn.reshape(-1))
    pmf_nf_first    = pmf_from(first_nf_syn.reshape(-1))
    pmf_bd_first    = pmf_from(first_basden_syn.reshape(-1))

    centers = 0.5 * (bins[:-1] + bins[1:])
    plt.figure(figsize=(10, 5))
    plt.plot(centers, pmf_bg,         label="Background (empirical)")
    plt.plot(centers, pmf_gauss_first, label="Gaussian model (sampled, trial 0)")
    plt.plot(centers, pmf_pg_first,    label="Poisson-Gaussian model (sampled, trial 0)")
    plt.plot(centers, pmf_nf_first,    label="NoiseFlow model (sampled, trial 0)")
    plt.plot(centers, pmf_bd_first,    label="Basden model (sampled, trial 0)")
    plt.xlabel("Raw intensity (DN)")
    plt.ylabel("Probability (pmf)")
    plt.title("Raw-domain PMF comparison (sample-based, first trial)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pmf_raw_bg_vs_models_sampled_first_trial.png"), dpi=200)
    plt.close()

    print(f"[saved synthetic background images & PMF plot] -> {out_dir}")
    
    # -----------------------------
    # 9) [New] PSD (Spatial Correlation) 비교
    # -----------------------------
    print("Computing Radial PSD...")
    
    # 첫 trial 이미지들 가져오기 (2D)
    img_real = arr[0]                   # Real Background
    img_nf   = first_nf_syn[0]  # NoiseFlow
    img_bd   = first_basden_syn[0]  # Basden (EMCCD)
    
    # 배경 평균 제거 (DC 성분 억제) - 패턴만 보기 위해
    img_real = img_real - img_real.mean()
    img_nf   = img_nf - img_nf.mean()
    img_bd   = img_bd - img_bd.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    
    plot_radial_psd(img_real, "Real Background", "black", ax)
    plot_radial_psd(img_nf,   "NoiseFlow (AI)",  "blue",  ax)
    plot_radial_psd(img_bd,   "Basden Model",    "red",   ax)
    # plot_radial_psd(img_pg,   "Poisson-Gauss",   "green", ax) # 너무 많으면 생략 가능

    ax.set_title("Spatial Structure Analysis (Radial PSD)")
    ax.set_xlabel("Spatial Frequency (Radius)")
    ax.set_ylabel("Power Spectrum (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spatial_psd_comparison.png"), dpi=200)
    plt.close()
    
    print(f"[saved PSD comparison plot] -> {out_dir}")
    
    N_test = 50 

    # 1. Real Images (이미 로드된 arr에서 가져옴)
    real_stack = arr[:N_test]

    # 2. Synthetic Images 생성 (함수 재사용)
    # (이미 정의된 sample 함수들 사용)
    print(f"Generating {N_test} synthetic frames for PSD averaging...")

    # NoiseFlow
    nf_stack = sample_nf_frames_from_mean(N_test)

    # Basden (EMCCD)
    # 주의: synthetic_mean_frame은 위에서 정의한 것 사용
    synthetic_mean_val = (0.0041 * 42.66) + 457.84 # lambda * gain + bias
    syn_mean_frame = np.full((64, 64), synthetic_mean_val, dtype=np.float32)

    bd_stack = sample_basden_emccd_frames(
        mean_frame_adu=syn_mean_frame,
        n_frames=N_test,
        em_gain=42.66,
        sigma_read=18.91,
        bias_offset=457.84
    )

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_average_psd(real_stack, "Real Background", "black", ax)
    plot_average_psd(nf_stack,   "NoiseFlow (AI)",  "blue",  ax)
    plot_average_psd(bd_stack,   "Basden Model",    "red",   ax)

    ax.set_title(f"Average Spatial PSD (N={N_test})")
    ax.set_xlabel("Spatial Frequency (Radius)")
    ax.set_ylabel("Power Spectrum (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "averaged_spatial_psd.png"), dpi=200)
    plt.close()
    print("Saved averaged_spatial_psd.png")
    

if __name__ == "__main__":
    # compare_gaussian_noise_model()
    compare_poisson_gaussian_noise_model()