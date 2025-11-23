import os
import glob
import types
import torch
import numpy as np

from tifffile import imwrite

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
    hps.flow_permutation = 1  # ← arg_parser 기본값이 뭔지 확인해서 필요하면 수정
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
        
@torch.no_grad()
def denoise_tif(
    model: Noise2NoiseFlow,
    hps,
    noisy_tif_path: str,
    out_tif_path: str = None,
    to_raw_scale: bool = True,
    save_single: bool = True,
):
    """
    단일 TIFF noisy 이미지를 로드 → 모델로 denoise.
    - save_single=True면 개별 파일로 저장
    - False면 저장하지 않고 배열만 리턴 (스택 만들 때 사용)
    """
    noisy = _load_tif_atom(noisy_tif_path)    # float32, [0,1], (C?,H,W)
    C = hps.x_shape[1]                        # 2
    noisy = _ensure_channels(noisy, C=C)      # (2,H,W)

    noisy_t = noisy.unsqueeze(0).to(hps.device)  # (1,2,H,W)
    denoised = model.denoise(noisy_t).clamp(0, 1)   # (1,2,H,W)

    denoised_np = denoised.squeeze(0).cpu().numpy() # (2,H,W)
    noisy_np    = noisy.cpu().numpy()               # (2,H,W)

    denoised_norm = denoised_np[0]  # (H,W), [0,1]
    noisy_norm    = noisy_np[0]

    if to_raw_scale:
        denoised_raw = denoised_norm * (GLOBAL_VMAX - GLOBAL_VMIN) + GLOBAL_VMIN
        noisy_raw    = noisy_norm    * (GLOBAL_VMAX - GLOBAL_VMIN) + GLOBAL_VMIN

        denoised_raw = np.clip(denoised_raw, 0, 65535)
        noisy_raw    = np.clip(noisy_raw, 0, 65535)

        save_arr = denoised_raw.astype(np.uint16)
        diff_max = np.abs(denoised_raw - noisy_raw).max()
    else:
        save_arr = (denoised_norm * 65535.0).clip(0, 65535).astype(np.uint16)
        noisy_raw = (noisy_norm * 65535.0).clip(0, 65535)
        diff_max = np.abs(save_arr.astype(np.float32) - noisy_raw).max()

    if save_single:
        if out_tif_path is None:
            base, ext = os.path.splitext(noisy_tif_path)
            out_tif_path = base + "_denoised.tif"

        imwrite(out_tif_path, save_arr)
        print(f"[saved] {noisy_tif_path} -> {out_tif_path}, max abs diff={diff_max}")
    else:
        # 스택 저장용이라면 출력만 조용히
        pass

    # 스택 만들 때 쓰라고 normalized(또는 raw) 반환
    return denoised_norm  # 필요하면 denoised_raw로 바꿔도 OK


@torch.no_grad()
def denoise_and_stack_a_only(
    model: Noise2NoiseFlow,
    hps,
    test_root: str,
    out_tif_path: str,
    to_raw_scale: bool = True,
    max_scenes: int | None = None,   # 일부만 테스트하고 싶으면 값 넣기
):
    """
    test_root/scene_xxxx/a.tif 들만 디노이즈해서
    (N, H, W) 스택으로 만들어 out_tif_path 하나에 저장.
    """
    pattern = os.path.join(test_root, 'scene_*')
    scene_dirs = sorted(glob.glob(pattern))

    if not scene_dirs:
        print(f"No scenes found under {test_root}")
        return

    denoised_list = []

    for i, scene in enumerate(scene_dirs):
        if max_scenes is not None and i >= max_scenes:
            break

        a_path = os.path.join(scene, 'a.tif')
        if not os.path.isfile(a_path):
            continue

        # 개별 파일은 저장하지 않고, 배열만 리턴받기
        denoised_norm = denoise_tif(
            model,
            hps,
            a_path,
            out_tif_path=None,
            to_raw_scale=False,   # 스택에서 한 번에 raw 스케일로 변환할 예정
            save_single=False,
        )
        denoised_list.append(denoised_norm)

    if not denoised_list:
        print("No a.tif files found to denoise.")
        return

    # (N, H, W) 스택
    stack_norm = np.stack(denoised_list, axis=0)  # float32, [0,1]

    if to_raw_scale:
        stack_raw = stack_norm * (GLOBAL_VMAX - GLOBAL_VMIN) + GLOBAL_VMIN
        stack_raw = np.clip(stack_raw, 0, 65535).astype(np.uint16)
        save_arr = stack_raw
    else:
        save_arr = (stack_norm * 65535.0).clip(0, 65535).astype(np.uint16)

    imwrite(out_tif_path, save_arr)
    print(f"[stack saved] {out_tif_path}  shape={save_arr.shape} (N,H,W)")
    
from tifffile import imread, imwrite

@torch.no_grad()
def denoise_background_file(
    model: Noise2NoiseFlow,
    hps,
    bg_tif_path: str,
    out_tif_path: str,
    to_raw_scale: bool = True,
):
    """
    background.tif 같은 단일 파일을 디노이즈해서 다시 TIF로 저장.
    - 입력: (H,W) 또는 (N,H,W) uint16
    - 출력: 같은 프레임 수/해상도로 디노이즈된 TIF
    """
    # --------------------------
    # 0) TIFF 읽기
    # --------------------------
    arr = imread(bg_tif_path)   # (H,W) 또는 (N,H,W)
    arr = arr.astype(np.float32)

    # (N,H,W) 형태로 통일
    if arr.ndim == 2:
        arr = arr[None, ...]    # (1,H,W)
    elif arr.ndim == 3:
        pass                    # 이미 (N,H,W)
    else:
        raise ValueError(f"Unsupported background shape: {arr.shape}")

    N, H, W = arr.shape
    print(f"[bg] loaded {bg_tif_path}, shape={arr.shape}, dtype=float32")

    # --------------------------
    # 1) 학습 때와 동일한 정규화: (x - VMIN) / (VMAX - VMIN)
    # --------------------------
    denom = float(GLOBAL_VMAX - GLOBAL_VMIN)
    if denom <= 0:
        raise ValueError(f"Invalid GLOBAL_VMIN/VMAX: {GLOBAL_VMIN}, {GLOBAL_VMAX}")

    arr_norm = (arr - GLOBAL_VMIN) / denom
    arr_norm = np.clip(arr_norm, 0.0, 1.0).astype(np.float32)   # (N,H,W), [0,1]

    denoised_norm_list = []

    # --------------------------
    # 2) 각 프레임별로 모델에 통과
    # --------------------------
    for i in range(N):
        frame = arr_norm[i]   # (H,W)

        # (1,H,W) → _ensure_channels로 C채널 맞추기
        frame_t = torch.from_numpy(frame).unsqueeze(0)     # (1,H,W)
        frame_t = _ensure_channels(frame_t, C=hps.x_shape[1])  # (C,H,W), C=2
        frame_t = frame_t.unsqueeze(0).to(hps.device)      # (1,C,H,W)

        den = model.denoise(frame_t).clamp(0, 1)           # (1,C,H,W)
        den_np = den.squeeze(0).cpu().numpy()              # (C,H,W)

        # 첫 채널만 사용 (background이니까 채널 의미 크게 없을 것)
        denoised_norm = den_np[0]                          # (H,W), [0,1]
        denoised_norm_list.append(denoised_norm)

    # (N,H,W) 스택
    stack_norm = np.stack(denoised_norm_list, axis=0)      # (N,H,W)

    # --------------------------
    # 3) 역정규화 + 저장
    # --------------------------
    if to_raw_scale:
        stack_raw = stack_norm * denom + GLOBAL_VMIN
        stack_raw = np.clip(stack_raw, 0, 65535).astype(np.uint16)
        save_arr = stack_raw
    else:
        # 보기용으로만 [0,1]→[0,65535] 매핑
        save_arr = (stack_norm * 65535.0).clip(0, 65535).astype(np.uint16)

    # 프레임이 1장인 경우 (1,H,W) → (H,W)로 줄여도 상관 없음
    if save_arr.shape[0] == 1:
        save_arr_to_write = save_arr[0]   # (H,W)
    else:
        save_arr_to_write = save_arr      # (N,H,W), 멀티페이지 TIF

    imwrite(out_tif_path, save_arr_to_write)
    print(f"[bg saved] {bg_tif_path} -> {out_tif_path}, shape={save_arr_to_write.shape}")

# ----------------------
# 5) 스크립트로 사용할 때
# ----------------------
# if __name__ == '__main__':
#     ckpt_path = 'experiments/weights/best_model_real.pth'
#     test_root = './data_atom/test'

#     model, hps = load_trained_model(ckpt_path, device='cuda')

#     # (1) 필요하면 여전히 개별 파일 저장
#     # denoise_test_folder(model, hps, test_root)

#     # (2) a.tif만 모아서 하나의 멀티페이지 TIFF로 저장
#     out_stack_path = './data_atom/denoised_a_stack.tif'
#     denoise_and_stack_a_only(
#         model,
#         hps,
#         test_root,
#         out_tif_path=out_stack_path,
#         to_raw_scale=True,
#         max_scenes=None,   # 예: 100으로 두면 처음 100 scene만
#     )
    
if __name__ == '__main__':
    ckpt_path = 'experiments/weights/best_model_real.pth'

    model, hps = load_trained_model(ckpt_path, device='cuda')

    # (1) 원래 test scenes 디노이즈
    # denoise_test_folder(model, hps, test_root)

    # (2) background 한 파일 디노이즈
    bg_in  = './data_atom/background.tif'
    bg_out = './data_atom/background_denoised.tif'
    denoise_background_file(model, hps, bg_in, bg_out, to_raw_scale=True)