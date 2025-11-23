import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile import imread
import os

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
    
def compute_pmf_from_tiff_stack(
    tiff_path,
    num_bins=200,
    subtract_mean=True,
    roi=None,
):
    """
    tiff_path: path to TIFF stack (T, H, W)
    num_bins: number of histogram bins
    subtract_mean: subtract overall mean to isolate noise
    roi: (y0, y1, x0, x1). If None, use entire image.

    Returns:
        bins: (num_bins+1,) array of bin boundaries
        pmf:  (num_bins,) normalized probability mass array
    """

    # ---- 1) TIFF 읽기 ----
    imgs = tiff.imread(tiff_path)   # shape (T, H, W)
    imgs = imgs.astype(np.float32)

    if imgs.ndim != 3:
        raise ValueError("TIFF must be a stack: shape should be (T, H, W).")

    T, H, W = imgs.shape

    # ---- 2) ROI 선택 ----
    if roi is not None:
        y0, y1, x0, x1 = roi
        imgs = imgs[:, y0:y1, x0:x1]  # shape: (T, Hy, Wx)

    # ---- 3) 평균 제거 (noise isolation) ----
    if subtract_mean:
        mean_val = imgs.mean()
        imgs = imgs - mean_val

    # ---- 4) 모든 픽셀을 1D 샘플로 flatten ----
    samples = imgs.reshape(-1)   # shape (T * Hy * Wx,)

    # ---- 5) bin 구간 정의 ----
    vmin, vmax = samples.min(), samples.max()
    bins = np.linspace(vmin, vmax, num_bins + 1)

    # ---- 6) 히스토그램 계산 ----
    hist, _ = np.histogram(samples, bins=bins, density=False)

    # ---- 7) PMF로 정규화 ----
    pmf = hist / hist.sum()

    return bins, pmf


def visualize_pmf(bins, pmf, title="PMF of Noise Distribution"):
    """
    bins: array of bin boundaries, shape (K+1,)
    pmf: array of probabilities, shape (K,)
    """

    centers = 0.5 * (bins[:-1] + bins[1:])  # bin 중심값

    plt.figure(figsize=(10, 5))

    # ---- 1) 라인 플롯 ----
    plt.plot(centers, pmf, '-o', markersize=2, label='PMF (line)')
    
    # ---- 2) 막대 플롯 ----
    plt.bar(centers, pmf, width=np.diff(bins), alpha=0.3, label='PMF (bar)')
    
    # ---- 3) 스텝 플롯 ----
    plt.step(bins[:-1], pmf, where='post', color='red', alpha=0.5, label='PMF (step)')

    plt.xlabel("Noise / Pixel Value")
    plt.ylabel("Probability (pmf)")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    
def visualize_pmf_pdf(bins, pmf, pdf, title="Noise PMF/PDF"):
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
    plt.show()
    
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
    
def main():
    ckpt_path = "experiments/weights/best_model_real.pth"
    bg_stack_path = "./data_atom/background.tif"  # (N,H,W) 또는 (H,W)

    # 1) 모델 로드 (flow 포함)
    model, hps = load_trained_model_for_flow(ckpt_path, device="cuda")

    # 2) bg mean 텐서 만들기 (clean background, [0,1] 스케일)
    bg_t, denom = build_bg_mean_tensor(bg_stack_path, hps)  # bg_t: (1,C,H,W)

    # 3) noise model에서 noisy background 샘플 여러 장 생성
    noisy_norm, noise_norm = sample_noisy_bg_from_flow(
        model,
        bg_t,
        n_samples=64,   # pmf 안정 위해 적당히 크게
    )

    # 4) noise 분포 pmf / pdf 계산 (노이즈만 보고 싶으면 noise_norm 사용)
    bins, pmf, pdf = compute_pmf_pdf_from_images(
        noise_norm,     # 또는 noisy_norm 사용 가능
        num_bins=200,
        subtract_mean=True,
    )

    # 5) pmf/pdf 시각화
    visualize_pmf_pdf(bins, pmf, pdf, title="NoiseFlow noise PMF/PDF")

    # 6) bg_mean / noise / noisy 이미지 저장
    save_example_images(
        bg_t=bg_t,
        noise_norm=noise_norm,
        noisy_norm=noisy_norm,
        denom=denom,
        out_dir="./noiseflow_viz",
        max_examples=1,
        
    )
    save_background_sample(bg_stack_path, out_dir="./noiseflow_viz")

    # 7) 간단한 수치 확인
    print("bins shape:", bins.shape)
    print("pmf shape :", pmf.shape)
    print("sum pmf   :", pmf.sum())
    print("pdf int ~ :", np.sum(pdf * np.diff(bins)))  # ≈ 1이면 정상


if __name__ == "__main__":
    main()