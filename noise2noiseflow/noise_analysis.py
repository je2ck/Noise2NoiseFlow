import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile import imread
import os
from scipy.stats import norm, poisson


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
   
    
def compare_poisson_gaussian_noise_model():
    ckpt_path = "experiments/weights/best_model_real.pth"
    bg_stack_path = "./data_atom/background.tif"  # (T,H,W) 또는 (H,W)
    out_dir = "./noiseflow_viz"
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # 1) 실제 background raw 스택 로드
    # -----------------------------
    arr = imread(bg_stack_path).astype(np.float32)  # (T,H,W) 또는 (H,W)
    if arr.ndim == 2:
        arr = arr[None, ...]  # (1,H,W)
    T, H, W = arr.shape

    # 전체 raw 샘플 (ground-truth)
    bg_samples_raw = arr.reshape(-1)

    # "clean" 역할을 할 mean frame (모든 프레임 평균)
    mean_frame = arr.mean(axis=0)   # (H,W)

    # -----------------------------
    # 2) 전역 통계로 Gaussian / Poisson-Gaussian 파라미터 추정
    # -----------------------------
    mean_bg = bg_samples_raw.mean()
    var_bg  = bg_samples_raw.var()

    # Gaussian: N( mean_bg, var_bg ) 라고 가정하지만
    # synthetic에서는 mean_frame + N(0, sigma_g^2) 형태로 사용할 거라
    # residual의 sigma만 쓰고, 평균은 mean_frame이 담당
    sigma_g = np.sqrt(max(var_bg, 1e-12))

    # Poisson-Gaussian: X = P(lam) + N(0, sigma_pg^2)
    lam_pg  = max(mean_bg, 1e-6)
    sigma2_pg = max(var_bg - lam_pg, 1e-12)   # Var(X) = lam + sigma^2 가정
    sig_pg  = np.sqrt(sigma2_pg)

    # -----------------------------
    # 3) NoiseFlow 로드
    # -----------------------------
    model, hps = load_trained_model_for_flow(ckpt_path, device="cuda")
    denom = float(GLOBAL_VMAX - GLOBAL_VMIN)

    @torch.no_grad()
    def sample_nf_frames_from_mean(n_frames):
        """
        mean_frame을 clean condition으로 두고
        NoiseFlow residual을 여러 장 샘플 → raw-domain synthetic frame 생성
        """
        frames = []

        # clean_norm: (1, C, H, W)
        clean_norm = (mean_frame - GLOBAL_VMIN) / denom
        clean_norm = np.clip(clean_norm, 0.0, 1.0).astype(np.float32)
        clean_t = torch.from_numpy(clean_norm).unsqueeze(0)        # (1,H,W)
        clean_t = _ensure_channels(clean_t, C=hps.x_shape[1])      # (C,H,W)
        clean_t = clean_t.unsqueeze(0).to(hps.device)              # (1,C,H,W)

        for _ in range(n_frames):
            # NoiseFlow residual (정규화 스케일)
            eps = model.noise_flow.sample(clean=clean_t)           # (1,C,H,W)
            eps_raw = eps[:, 0].detach().cpu().numpy()[0] * denom  # (H,W) raw residual

            frame_raw = mean_frame + eps_raw                       # (H,W) raw synthetic
            frames.append(frame_raw)

        return np.stack(frames, axis=0)   # (n_frames, H, W)

    # -----------------------------
    # 4) 세 가지 모델에서 synthetic frame 샘플링
    # -----------------------------
    Nf = 512   # 각 모델당 생성할 프레임 수

    # (a) NoiseFlow
    nf_syn = sample_nf_frames_from_mean(Nf)         # (Nf,H,W)
    nf_samples_raw = nf_syn.reshape(-1)

    # (b) Gaussian: mean_frame + N(0, sigma_g^2)
    gauss_residual = np.random.normal(
        loc=0.0, scale=sigma_g, size=(Nf, H, W)
    ).astype(np.float32)
    gauss_syn = mean_frame[None, ...] + gauss_residual
    gauss_samples_raw = gauss_syn.reshape(-1)

    # (c) Poisson-Gaussian: mean_frame + [ (P(lam) - lam) + N(0, sig_pg^2) ]
    S = np.random.poisson(lam=lam_pg, size=(Nf, H, W)).astype(np.float32)
    G = np.random.normal(loc=0.0, scale=sig_pg, size=(Nf, H, W)).astype(np.float32)
    pg_residual = (S - lam_pg) + G
    pg_syn = mean_frame[None, ...] + pg_residual
    pg_samples_raw = pg_syn.reshape(-1)

    # -----------------------------
    # 5) 공통 bins 만들고 PMF & KL 계산
    # -----------------------------
    num_bins = 200
    all_samples = np.concatenate(
        [bg_samples_raw, gauss_samples_raw, pg_samples_raw, nf_samples_raw],
        axis=0
    )
    vmin, vmax = all_samples.min(), all_samples.max()
    bins = np.linspace(vmin, vmax, num_bins + 1)

    def pmf_from(samples):
        hist, _ = np.histogram(samples, bins=bins, density=False)
        pmf = hist / hist.sum()
        return pmf

    pmf_bg    = pmf_from(bg_samples_raw)
    pmf_gauss = pmf_from(gauss_samples_raw)
    pmf_pg    = pmf_from(pg_samples_raw)
    pmf_nf    = pmf_from(nf_samples_raw)

    KL_gauss = kl_divergence_pmf(bins, pmf_bg, bins, pmf_gauss)
    KL_pg    = kl_divergence_pmf(bins, pmf_bg, bins, pmf_pg)
    KL_nf    = kl_divergence_pmf(bins, pmf_bg, bins, pmf_nf)

    print("=== Raw-domain KL divergences (background || model) ===")
    print(f"KL(background || Gaussian)         = {KL_gauss}")
    print(f"KL(background || Poisson-Gaussian) = {KL_pg}")
    print(f"KL(background || NoiseFlow)        = {KL_nf}")

    # -----------------------------
    # 6) synthetic background 예시 이미지 저장
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

    # (b) Gaussian model synthetic
    _save_raw_image(gauss_syn[0],
                    os.path.join(out_dir, "bg_gaussian_model_raw.png"),
                    "Gaussian model synthetic background")

    # (c) Poisson-Gaussian model synthetic
    _save_raw_image(pg_syn[0],
                    os.path.join(out_dir, "bg_poisson_gaussian_model_raw.png"),
                    "Poisson-Gaussian model synthetic background")

    # (d) NoiseFlow model synthetic
    _save_raw_image(nf_syn[0],
                    os.path.join(out_dir, "bg_noiseflow_model_raw.png"),
                    "NoiseFlow model synthetic background")

    # -----------------------------
    # 7) PMF 비교 플롯
    # -----------------------------
    centers = 0.5 * (bins[:-1] + bins[1:])
    plt.figure(figsize=(10, 5))
    plt.plot(centers, pmf_bg,    label="Background (empirical)")
    plt.plot(centers, pmf_gauss, label="Gaussian model (sampled)")
    plt.plot(centers, pmf_pg,    label="Poisson-Gaussian model (sampled)")
    plt.plot(centers, pmf_nf,    label="NoiseFlow model (sampled)")
    plt.xlabel("Raw intensity (DN)")
    plt.ylabel("Probability (pmf)")
    plt.title("Raw-domain PMF comparison (sample-based)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pmf_raw_bg_vs_models_sampled.png"), dpi=200)
    plt.close()

    print(f"[saved synthetic background images & PMF plot] -> {out_dir}")
    
    
    
if __name__ == "__main__":
    compare_gaussian_noise_model()
    compare_poisson_gaussian_noise_model()