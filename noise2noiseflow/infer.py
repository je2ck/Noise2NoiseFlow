import os
import glob
import types
import torch
import numpy as np

import sys
sys.path.append('../')

from model.noise2noise_flow import Noise2NoiseFlow
from train_atom import init_params, _load_tif, _load_tif_atom, _ensure_channels  # 네가 올린 학습 파일 기준
from train_atom import GLOBAL_VMIN, GLOBAL_VMAX

# ----------------------
# 1) 하이퍼파라미터 준비
# ----------------------
def build_hps(device='cuda'):
    hps = types.SimpleNamespace()

    # 학습 때와 동일하게 맞추기
    hps.arch = "unc|unc|unc|unc|gain|unc|unc|unc|unc"
    hps.flow_permutation = 'conv'   # ← arg_parser 기본값이 뭔지 확인해서 필요하면 수정
    hps.lu_decomp = True
    hps.denoiser = 'dncnn'          # 학습 때도 dncnn 썼다면 그대로
    hps.lmbda = 262144

    # 실제 학습에서 쓰인 입력 shape
    C = 2        # 중요: dataset이 최소 2채널로 맞췄음
    H, W = 64,64
    hps.x_shape = (1, C, H, W)   # (B,C,H,W)

    hps.device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    return hps

# ----------------------
# 2) 모델 로드 함수
# ----------------------
def load_trained_model(ckpt_path: str, device='cuda'):
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
    full_state = ckpt['state_dict']

    # 1) 현재 모델 state
    model_state = model.state_dict()
    # 2) denoiser.* 키만 필터링해서, 이름과 shape이 맞는 것만 덮어쓰기
    for k, v in full_state.items():
        if k.startswith("denoiser.") and k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v

    model.load_state_dict(model_state, strict=False)

    model.to(hps.device)
    model.eval()
    return model, hps

# ----------------------
# 3) 단일 TIFF 이미지 denoise
# ----------------------
@torch.no_grad()
def denoise_tif(model: Noise2NoiseFlow, hps, noisy_tif_path: str, out_tif_path: str = None,
                to_raw_scale: bool = True):
    """
    단일 TIFF noisy 이미지를 로드 → 모델로 denoise → 저장.
    - 학습 시 (x - ATOM_VMIN) / (ATOM_VMAX - ATOM_VMIN) 으로 정규화했다면,
      저장 시에는 역변환해서 원래 카메라 count 스케일로 복원.
    """
    # --------------------------
    # 0) Load TIFF as float32 (C,H,W), [0,1] normalized
    # --------------------------
    noisy = _load_tif_atom(noisy_tif_path)    # float32, [0,1], (C?,H,W)
    C = hps.x_shape[1]                        # 2
    noisy = _ensure_channels(noisy, C=C)      # (2,H,W)

    # --------------------------
    # 1) Move to device + batch dim
    # --------------------------
    noisy_t = noisy.unsqueeze(0).to(hps.device)  # (1,2,H,W)

    # --------------------------
    # 2) Run denoiser (normalized space)
    # --------------------------
    denoised = model.denoise(noisy_t).clamp(0, 1)   # (1,2,H,W)

    # --------------------------
    # 3) Convert to numpy (normalized)
    # --------------------------
    denoised_np = denoised.squeeze(0).cpu().numpy() # (2,H,W)
    noisy_np    = noisy.cpu().numpy()               # (2,H,W)

    # --------------------------
    # 4) Pick channel for saving (여기선 ch0)
    # --------------------------
    denoised_norm = denoised_np[0]  # (H,W), [0,1]
    noisy_norm    = noisy_np[0]     # (H,W), [0,1]

    # --------------------------
    # 5) 역정규화 (옵션)
    # --------------------------
    if to_raw_scale:
        # [0,1] -> raw 카메라 count 스케일
        denoised_raw = denoised_norm * (GLOBAL_VMAX- GLOBAL_VMIN) + GLOBAL_VMIN
        noisy_raw    = noisy_norm * (GLOBAL_VMAX - GLOBAL_VMIN) + GLOBAL_VMIN

        # uint16 범위로 클리핑
        denoised_raw = np.clip(denoised_raw, 0, 65535)
        noisy_raw    = np.clip(noisy_raw, 0, 65535)

        save_arr = denoised_raw.astype(np.uint16)
        diff_max = np.abs(denoised_raw - noisy_raw).max()
    else:
        # 단순히 [0,1]을 16bit full range로 맵핑해서 보기용으로만 저장
        save_arr = (denoised_norm * 65535.0).clip(0, 65535).astype(np.uint16)
        noisy_raw = (noisy_norm * 65535.0).clip(0, 65535)
        diff_max = np.abs(save_arr.astype(np.float32) - noisy_raw).max()

    # --------------------------
    # 6) Save TIFF
    # --------------------------
    if out_tif_path is None:
        base, ext = os.path.splitext(noisy_tif_path)
        out_tif_path = base + "_denoised.tif"

    from tifffile import imwrite
    imwrite(out_tif_path, save_arr)

    print(f"[saved] {noisy_tif_path} -> {out_tif_path}, max abs diff={diff_max}")

    return denoised_norm  # 필요하면 denoised_raw로 바꿔도 됨



# ----------------------
# 4) test 폴더 전체 돌리기
# ----------------------
@torch.no_grad()
def denoise_test_folder(model: Noise2NoiseFlow, hps, test_root: str):
    """
    test_root/
      scene_000/
        a.tif
        b.tif
      scene_001/
        a.tif
        b.tif
      ...
    이런 구조라고 가정.
    """
    pattern = os.path.join(test_root, 'scene_*')
    scene_dirs = sorted(glob.glob(pattern))

    if not scene_dirs:
        print(f"No scenes found under {test_root}")
        return

    print(f"Found {len(scene_dirs)} scenes under {test_root}")
    i = 0
    for scene in scene_dirs:
        if i > 2:
            break
        for name in ['a.tif', 'b.tif']:
            tif_path = os.path.join(scene, name)
            if os.path.isfile(tif_path):
                denoise_tif(model, hps, tif_path)
            # b.tif는 필요 없으면 위 루프에서 'a.tif'만 돌려도 됨
        i += 1

# ----------------------
# 5) 스크립트로 사용할 때
# ----------------------
if __name__ == '__main__':
    # 학습 때 저장된 best 모델 경로
    ckpt_path = 'experiments/weights/best_model_real.pth'

    # test 폴더 루트 (scene_*들이 있는 폴더)
    test_root = './data_atom/test'   # 예: 'data/custom_dataset/test' 이런 식으로

    model, hps = load_trained_model(ckpt_path, device='cuda')
    denoise_test_folder(model, hps, test_root)