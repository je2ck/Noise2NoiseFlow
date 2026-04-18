import os
import types
import torch
import numpy as np
import argparse
from tifffile import imread, imwrite
import sys

# 프로젝트 구조에 따라 경로 추가
sys.path.append('../')

from model.noise2noise_flow import Noise2NoiseFlow
from train_atom import init_params, _load_tif, _load_tif_atom, _ensure_channels

# ----------------------
# 1) 하이퍼파라미터 준비
# ----------------------
def build_hps(args, device='cuda'):
    hps = types.SimpleNamespace()
    if args.basden:
        hps.arch = "basden|sds"
        hps.basden_config = {
            'vmin': args.vmin, 'vmax': args.vmax,
            'bias_offset': args.basden_bias_offset,
            'readout_sigma': args.basden_readout_sigma,
            'em_gain': args.basden_em_gain,
            'sensitivity': args.basden_sensitivity,
            'cic_lambda': args.basden_cic_lambda,
        }
    else:
        hps.arch = "unc|unc|unc|unc|gain|unc|unc|unc|unc"
        hps.basden_config = None

    hps.flow_permutation = 1
    hps.lu_decomp = True
    hps.denoiser = 'dncnn'
    hps.lmbda = 262144
    hps.x_shape = (1, 2, 64, 64) # C=2, H=64, W=64
    hps.device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    hps.vmin = args.vmin
    hps.vmax = args.vmax
    return hps

# ----------------------
# 2) 모델 로드 함수
# ----------------------
def load_trained_model(args, ckpt_path: str, device='cuda'):
    hps = build_hps(args, device=device)
    param_inits = init_params()
    model = Noise2NoiseFlow(
        hps.x_shape[1:], arch=hps.arch, flow_permutation=hps.flow_permutation,
        param_inits=param_inits, lu_decomp=hps.lu_decomp, basden_config=hps.basden_config,
        denoiser_model=hps.denoiser, dncnn_num_layers=9, lmbda=hps.lmbda, device=hps.device
    )
    ckpt = torch.load(ckpt_path, map_location=hps.device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(hps.device)
    model.eval()
    return model, hps

# ----------------------
# 3) 범용 디노이징 함수 (파일 단위)
# ----------------------
@torch.no_grad()
def denoise_tif_file(model, hps, input_path, output_path, to_raw_scale=True):
    if not input_path or not os.path.exists(input_path):
        return

    arr = imread(input_path).astype(np.float32)
    if arr.ndim == 2: arr = arr[None, ...]
    N, H, W = arr.shape
    print(f"[*] Processing: {input_path} ({N} frames)")

    denom = float(hps.vmax - hps.vmin)
    arr_norm = np.clip((arr - hps.vmin) / denom, 0.0, 1.0) # 정규화

    denoised_list = []
    for i in range(N):
        frame_t = torch.from_numpy(arr_norm[i]).unsqueeze(0)
        frame_t = _ensure_channels(frame_t, C=hps.x_shape[1]) # 채널 확장
        frame_t = frame_t.unsqueeze(0).to(hps.device)

        den = model.denoise(frame_t).clamp(0, 1)
        denoised_list.append(den.squeeze(0).cpu().numpy()[0]) # 첫 채널 추출

    stack_norm = np.stack(denoised_list, axis=0)
    if to_raw_scale:
        save_arr = np.clip(stack_norm * denom + hps.vmin, 0, 65535).astype(np.uint16)
    else:
        save_arr = (stack_norm * 65535.0).clip(0, 65535).astype(np.uint16)

    if save_arr.shape[0] == 1: save_arr = save_arr[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imwrite(output_path, save_arr)
    print(f"[+] Saved: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 데이터 입력
    parser.add_argument('--input', type=str, required=True, help='Main noisy TIF')
    parser.add_argument('--output', type=str, required=True, help='Main output path')
    parser.add_argument('--bg_input', type=str, help='Optional background TIF')
    parser.add_argument('--bg_output', type=str, help='Optional background output path')
    parser.add_argument('--ckpt', type=str, required=True)
    
    # 물리 파라미터
    parser.add_argument('--vmin', type=float, default=384.0)
    parser.add_argument('--vmax', type=float, default=634.0)
    parser.add_argument('--basden', action='store_true', default=True)
    parser.add_argument('--basden_bias_offset', type=float, default=440.81)
    parser.add_argument('--basden_readout_sigma', type=float, default=19.69)
    parser.add_argument('--basden_em_gain', type=float, default=300.0)
    parser.add_argument('--basden_sensitivity', type=float, default=4.15)
    parser.add_argument('--basden_cic_lambda', type=float, default=0.0574)
    args = parser.parse_args()

    model, hps = load_trained_model(args, args.ckpt)

    # (1) 메인 파일 디노이징
    denoise_tif_file(model, hps, args.input, args.output)

    # (2) 백그라운드 파일 디노이징 (있을 경우만)
    if args.bg_input and args.bg_output:
        denoise_tif_file(model, hps, args.bg_input, args.bg_output)