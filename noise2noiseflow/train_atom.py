import glob
import logging
import os
import random
import re
import shutil
import socket
import time
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')

from utils.arg_parser import arg_parser
from data_loader.utils import ResultLogger
from utils.mylogger import add_logging_level
from utils.patch_stats_calculator import PatchStatsCalculator
from model.noise2noise_flow import Noise2NoiseFlow
from data_loader.sidd_utils import calc_kldiv_mb

try:
    import tifffile  # type: ignore
except ImportError:  # pragma: no cover
    tifffile = None


def _is_custom_dataset(root: str) -> bool:
    return os.path.isdir(os.path.join(root, 'train')) and os.path.isdir(os.path.join(root, 'val'))


def _load_tif(path: str) -> torch.Tensor:
    if tifffile is not None:
        array = tifffile.imread(path)
    else:
        with Image.open(path) as img:
            array = np.array(img)
    logging.debug("Loaded TIFF %s with shape %s and dtype %s", path, array.shape, array.dtype)
    if array.ndim == 2:
        array = array[:, :, None]          # (H, W) → (H, W, 1)
    elif array.ndim == 3:
        # If first dimension is small → probably channel-first (CHW)
        if array.shape[0] <= 4:
            array = np.transpose(array, (0, 1, 2))   # do nothing (CHW)
        else:
            array = np.transpose(array, (2, 0, 1))   # HWC → CHW
    else:
        raise ValueError(f"Unsupported TIFF shape: {array.shape}")
    dtype = array.dtype
    array = array.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if info.max > 0:
            array /= float(info.max)
    array = np.clip(array, 0.0, 1.0)
    array = np.transpose(array, (2, 0, 1))
    logging.debug("Normalized TIFF %s to tensor shape %s", path, array.shape)
    return torch.from_numpy(array)

GLOBAL_VMIN = 410   # 예: train 전체에서 1% 퍼센타일
GLOBAL_VMAX = 622  # 예: train 전체에서 99% 퍼센타일


def _load_tif_atom(path: str,
                   vmin: float = GLOBAL_VMIN,
                   vmax: float = GLOBAL_VMAX) -> torch.Tensor:
    """
    Atom imaging용 TIFF 로더 (global vmin/vmax 사용).

    - TIFF → float32
    - [vmin, vmax] 구간을 0~1로 선형 스케일링
    - 최종 shape: (C, H, W) 텐서
    """
    # 1) TIFF 읽기
    if tifffile is not None:
        array = tifffile.imread(path)
    else:
        with Image.open(path) as img:
            array = np.array(img)

    # 2) float32로 변환
    array = array.astype(np.float32)

    # 3) global vmin/vmax로 0~1 정규화
    if vmax <= vmin:
        raise ValueError(f"Invalid vmin/vmax: vmin={vmin}, vmax={vmax}")

    array = (array - vmin) / (vmax - vmin)
    array = np.clip(array, 0.0, 1.0)

    # 4) 채널 차원 정리 → (C, H, W)
    if array.ndim == 2:
        # (H, W) → (1, H, W)
        array = array[None, :, :]

    elif array.ndim == 3:
        # (H, W, C) 또는 (C, H, W) 둘 중 하나라고 가정
        # H, W가 같고 C가 작으면 둘 다 애매하니까, 다음 정도로 분기
        if array.shape[0] <= 4 and array.shape[1] == array.shape[2]:
            # (C, H, W) 형태라고 보고 그대로 사용
            pass
        elif array.shape[-1] <= 4 and array.shape[0] == array.shape[1]:
            # (H, W, C) → (C, H, W)
            array = np.transpose(array, (2, 0, 1))
        else:
            # 애매하면 일단 마지막 축을 채널로 보고 transpose
            array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported TIFF shape for atom loader: {array.shape}")

    logging.debug(
        "Atom-normalized %s -> shape=%s | final min=%.4f | max=%.4f",
        path, array.shape, float(array.min()), float(array.max())
    )

    return torch.from_numpy(array)

def _ensure_channels(x: torch.Tensor, C: int) -> torch.Tensor:
    """
    x: (C,H,W) 텐서. 1채널이면 C로 반복 복제, C보다 크면 앞 C개만 사용.
    """
    if x.dim() == 2:  # (H,W)면 (1,H,W)로
        x = x.unsqueeze(0)
    if x.size(0) == 1 and C > 1:
        x = x.repeat(C, 1, 1)
    elif x.size(0) > C:
        x = x[:C]
    return x


def _crop_pair(noisy: torch.Tensor, clean: torch.Tensor, patch: int, stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if patch is None:
        logging.debug("Stage %s using full image crop", stage)
        return noisy, clean
    _, h, w = noisy.shape
    if patch > h or patch > w:
        raise ValueError(f'Patch size {patch} exceeds image size {(h, w)}')
    if stage == 'train':
        top = random.randint(0, h - patch)
        left = random.randint(0, w - patch)
    else:
        top = (h - patch) // 2
        left = (w - patch) // 2
    logging.debug("Stage %s crop window top=%d left=%d size=%d", stage, top, left, patch)
    return noisy[:, top:top + patch, left:left + patch], clean[:, top:top + patch, left:left + patch]


class PairedTifPatchDataset(Dataset):
    def __init__(self, root_dir: str, stage: str, patch_size: int = None, patches_per_image: int = 1, desired_channels: int = 2):
        self.root_dir = root_dir
        self.stage = stage
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image or 1
        pattern = os.path.join(self.root_dir, 'scene_*')
        self.samples = []
        for scene in sorted(glob.glob(pattern)):
            noisy_path = os.path.join(scene, 'a.tif')
            clean_path = os.path.join(scene, 'b.tif')
            if os.path.isfile(noisy_path) and os.path.isfile(clean_path):
                self.samples.append((noisy_path, clean_path))
        if not self.samples:
            raise FileNotFoundError(f'No tif pairs found under {self.root_dir}')
        self.length = len(self.samples) * self.patches_per_image
        self.desired_channels = max(2, int(desired_channels))
        logging.info("Initialized %s dataset at %s with %d scenes (patches per image=%d)", stage, root_dir, len(self.samples), self.patches_per_image)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_idx = idx // self.patches_per_image
        noisy_path, clean_path = self.samples[scene_idx]
        if idx % max(self.patches_per_image, 1) == 0:
            logging.debug("Loading scene %d (%s, %s) for stage %s", scene_idx, noisy_path, clean_path, self.stage)
        noisy = _load_tif_atom(noisy_path)
        clean = _load_tif_atom(clean_path)
        noisy, clean = _crop_pair(noisy, clean, self.patch_size, self.stage)
        
        noisy = _ensure_channels(noisy, self.desired_channels)
        clean = _ensure_channels(clean, self.desired_channels)

        if self.stage in ('train', 'val'):
            return {
                'noisy1': noisy.contiguous(),
                'noisy2': clean.contiguous(),
            }

        noise = noisy - clean
        return {
            'noise': noise.contiguous(),
            'clean': clean.contiguous(),
            'pid': torch.tensor(idx % self.patches_per_image, dtype=torch.int32),
        }

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch_num: int, checkpoint_dir: str) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'epoch_num': epoch_num,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_dir)
    logging.info("Checkpoint saved to %s (epoch %d)", checkpoint_dir, epoch_num)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dir: str):
    """Load training checkpoint and restore model/optimizer state."""
    checkpoint = torch.load(checkpoint_dir, map_location=hps.device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info("Checkpoint loaded from %s (epoch %d)", checkpoint_dir, checkpoint['epoch_num'])
    return model, optimizer, checkpoint['epoch_num']

def init_params():
    """Initialize Noise2NoiseFlow camera/physics parameters."""
    npcam = 3
    c_i = 1.0
    beta1_i = -5.0 / c_i
    beta2_i = 0.0
    gain_params_i = np.ndarray([5])
    gain_params_i[:] = -5.0 / c_i
    cam_params_i = np.ndarray([npcam, 5])
    cam_params_i[:, :] = 1.0

    logging.debug(
        "Camera params initialized (c=%.3f, beta1=%.3f, beta2=%.3f)",
        c_i,
        beta1_i,
        beta2_i,
    )
    return (c_i, beta1_i, beta2_i, gain_params_i, cam_params_i)


def setup_logging() -> None:
    """Configure root logger with a readable format and custom DEBUG level."""
    try:
        add_logging_level('DEBUG', logging.DEBUG - 5)
    except AttributeError:
        # Level already exists; ignore
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("Logging initialized (level=%s)", logging.getLevelName(logging.getLogger().level))


def setup_device_and_seed(hps) -> None:
    """Set deterministic seeds and choose computation device; also set derived hparams."""
    torch.random.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    hps.n_bins = 2.0 ** hps.n_bits_x
    logging.debug('Num GPUs Available: %s' % torch.cuda.device_count())
    hps.device = 'cuda' if torch.cuda.device_count() else 'cpu'
    logging.info(
        "Seed set to %d | device=%s | n_bins=%.0f",
        hps.seed,
        hps.device,
        hps.n_bins,
    )


def prepare_logdirs(hps) -> None:
    """Prepare experiment directories and attach them to hps.

    Side effects:
      - hps.logdir is set to absolute experiments path
      - hps.logdirname retains the short name
      - Creates tensorboard and model directories
      - Optionally clears logdir when hps.no_resume is True
    """
    # experiment output dir (kept consistent with original: experiments/paper/<logdir>/)
    logdir = os.path.abspath(os.path.join('experiments', 'paper', hps.logdir)) + '/'

    if getattr(hps, 'no_resume', False):
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
            logging.warning("Removed existing logdir %s due to --no_resume", logdir)

    os.makedirs(logdir, exist_ok=True)
    hps.logdirname = hps.logdir
    hps.logdir = logdir
    logging.info("Experiment artifacts will be saved under %s", hps.logdir)

    # Attach frequently used sub-dirs
    hps.tensorboard_save_dir = os.path.join(hps.logdir, 'tensorboard_logs')
    hps.model_save_dir = os.path.join(hps.logdir, 'saved_models')
    os.makedirs(hps.tensorboard_save_dir, exist_ok=True)
    os.makedirs(hps.model_save_dir, exist_ok=True)
    logging.debug("TensorBoard dir: %s", hps.tensorboard_save_dir)
    logging.debug("Model ckpt dir: %s", hps.model_save_dir)


def is_validation_epoch(epoch: int, hps) -> bool:
    """Decide whether to run validation/tests this epoch (behaves like original condition)."""
    return epoch < 10 or (epoch < 100 and epoch % 10 == 0) or epoch % hps.epochs_full_valid == 0.0


# ------------------------------
# Data
# ------------------------------
def _build_custom_dataloaders(hps) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[int, ...]]:
    train_dir = os.path.join(hps.sidd_path, 'train')
    val_dir = os.path.join(hps.sidd_path, 'val')
    patch = getattr(hps, 'patch_height', None)
    patch = patch if patch else None
    patches_per_image = getattr(hps, 'n_patches_per_image', None) or 1
    num_workers = getattr(hps, 'n_train_threads', 0) or 0

    train_dataset = PairedTifPatchDataset(
        train_dir,
        stage='train',
        patch_size=patch,
        patches_per_image=patches_per_image,
    )
    val_dataset = PairedTifPatchDataset(
        val_dir,
        stage='val',
        patch_size=patch,
        patches_per_image=1,
    )
    test_dataset = PairedTifPatchDataset(
        val_dir,
        stage='eval',
        patch_size=patch,
        patches_per_image=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.n_batch_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=hps.n_batch_test,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hps.n_batch_test,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=True,
    )

    first_batch = next(iter(train_loader))
    x_shape = first_batch['noisy1'].shape
    hps.x_shape = x_shape
    hps.n_dims = int(np.prod(x_shape[1:]))
    hps.n_tr_inst = len(train_dataset.samples)
    hps.n_ts_inst = len(test_dataset.samples)
    hps.raw = False
    hps.n_channels = x_shape[1]

    logging.debug('# training scenes (custom) = {}'.format(hps.n_tr_inst))
    logging.debug('# validation scenes (custom) = {}'.format(len(val_dataset.samples)))
    logging.info(
        "Custom loaders ready | train=%d (patches=%d) | val=%d | test=%d | batch=%d/%d",
        hps.n_tr_inst,
        len(train_dataset),
        len(val_dataset.samples),
        len(test_dataset.samples),
        hps.n_batch_train,
        hps.n_batch_test,
    )

    return train_loader, validation_loader, test_loader, tuple(x_shape)


def build_dataloaders(hps) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[int, ...]]:
    return _build_custom_dataloaders(hps)

def compute_patch_stats_and_baselines(hps, test_loader: DataLoader, x_shape: Tuple[int, ...]):
    """Compute patch statistics and Gaussian/SDN baselines for reporting."""
    if getattr(hps, 'dataset_kind', '') == 'custom':
        logging.info('Skipping SIDD baseline stats for custom dataset')
        return {'sc_in_sd': None}, 0.0, 0.0

    logging.debug('calculating data stats and baselines...')
    pat_stats_calculator = PatchStatsCalculator(
        test_loader, x_shape[-1], n_channels=hps.n_channels, save_dir=hps.logdir, file_postfix=''
    )
    pat_stats = pat_stats_calculator.calc_stats()
    nll_gauss, nll_sdn = pat_stats_calculator.calc_baselines(test_loader)
    return pat_stats, nll_gauss, nll_sdn


# ------------------------------
# Model
# ------------------------------
def build_model_and_optimizer(hps) -> Tuple[Noise2NoiseFlow, torch.optim.Optimizer]:
    """Create model, optionally load a pretrained denoiser, move to device, and create optimizer."""
    hps.param_inits = init_params()
    model = Noise2NoiseFlow(
        hps.x_shape[1:],
        arch=hps.arch,
        flow_permutation=hps.flow_permutation,
        param_inits=hps.param_inits,
        lu_decomp=hps.lu_decomp,
        denoiser_model=hps.denoiser,
        dncnn_num_layers=9,
        lmbda=hps.lmbda,
        device=hps.device,
    )

    if getattr(hps, 'pretrained_denoiser', False):
        if hps.denoiser == 'dncnn':
            checkpoint_dir = '../denoisers/DnCNN_pretrained.pth'
            checkpoint = torch.load(checkpoint_dir, map_location=hps.device)
            model.denoiser.load_state_dict(checkpoint)
        elif hps.denoiser == 'unet':
            checkpoint_dir = '../denoisers/UNet_pretrained.pth'
            checkpoint = torch.load(checkpoint_dir, map_location=hps.device)
            model.denoiser.load_state_dict(checkpoint)

    model.to(hps.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hps.lr, betas=(0.9, 0.999), eps=1e-08)
    hps.num_params = int(np.sum([np.prod(params.shape) for params in model.parameters()]))
    print("noiseflow num params: {}".format(int(np.sum([np.prod(params.shape) for params in model.noise_flow.parameters()]))))
    print("Denoiser num params: {}".format(np.sum([np.prod(params.shape) for params in model.denoiser.parameters()])))
    logging.info(
        "Model ready | total params=%d | flow=%d | denoiser=%d | pretrained=%s",
        hps.num_params,
        int(np.sum([np.prod(p.shape) for p in model.noise_flow.parameters()])),
        int(np.sum([np.prod(p.shape) for p in model.denoiser.parameters()])),
        getattr(hps, 'pretrained_denoiser', False),
    )

    return model, optimizer


def try_resume(hps, model: Noise2NoiseFlow, optimizer: torch.optim.Optimizer) -> int:
    """Resume from the latest epoch_XXX checkpoint if available. Returns start_epoch."""
    # If directories were created by prepare_logdirs, they already exist.
    # Check for any epoch_*.pth files and resume from max epoch.
    start_epoch = 1
    entries = []
    if os.path.isdir(hps.model_save_dir):
        entries = [f for f in os.listdir(hps.model_save_dir) if os.path.isfile(os.path.join(hps.model_save_dir, f))]

    # Match files like 'epoch_123_nf_model_net.pth'
    epoch_re = re.compile(r'^epoch_(\d+)_nf_model_net\.pth$')
    epochs = []
    for name in entries:
        m = epoch_re.match(name)
        if m:
            epochs.append(int(m.group(1)))

    if epochs:
        last_epoch = max(epochs)
        saved_model_file_name = f'epoch_{last_epoch}_nf_model_net.pth'
        saved_model_file_path = os.path.join(hps.model_save_dir, saved_model_file_name)
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, saved_model_file_path)
        start_epoch = int(start_epoch) + 1
        logging.debug('found an existing previous checkpoint, resuming from epoch {}'.format(start_epoch))
    else:
        # No prior checkpoints
        start_epoch = 1
        logging.info("No checkpoint found in %s; starting fresh", hps.model_save_dir)

    return start_epoch


def create_loggers_and_writer(hps, start_epoch: int):
    """Create text loggers and TensorBoard writer."""
    log_columns = ['epoch', 'NLL', 'NLL_G', 'NLL_SDN']
    kld_columns = ['KLD_G', 'KLD_NLF', 'KLD_NF', 'KLD_R']
    train_logger = ResultLogger(os.path.join(hps.logdir, 'train.txt'), log_columns + ['train_time'], start_epoch > 1)
    validation_logger = ResultLogger(os.path.join(hps.logdir, 'validaton.txt'), log_columns, start_epoch > 1)
    test_logger = ResultLogger(os.path.join(hps.logdir, 'test.txt'), log_columns + ['msg'], start_epoch > 1)
    sample_logger = ResultLogger(os.path.join(hps.logdir, 'sample.txt'), log_columns + ['sample_time'] + kld_columns, start_epoch > 1)
    writer = SummaryWriter(hps.tensorboard_save_dir)
    logging.info("Result loggers and TensorBoard writer initialized at epoch %d", start_epoch)
    return train_logger, validation_logger, test_logger, sample_logger, writer


# ------------------------------
# Epoch routines
# ------------------------------
def _global_step_offset(dataset_len: int, batch_size: int, epoch: int, batch_index: int) -> int:
    steps_per_epoch = (dataset_len // batch_size) if batch_size > 0 else 0
    return int((epoch - 1) * steps_per_epoch + batch_index)


def train_one_epoch(model: Noise2NoiseFlow, optimizer: torch.optim.Optimizer, train_loader: DataLoader, hps, writer: SummaryWriter, epoch: int) -> Dict[str, float]:
    model.train()
    losses, nlls, mses = [], [], []
    start = time.time()

    num_batches = len(train_loader)
    logging.info("Epoch %d | train | batches=%d", epoch, num_batches)

    # Try to use dataset.__len__ if available; fall back to hps.n_tr_inst
    try:
        dataset_len = train_loader.dataset.__len__()
    except Exception:
        dataset_len = getattr(hps, 'n_tr_inst', 0)

    report_interval = max(1, num_batches // 5)

    for n_patch, image in enumerate(train_loader):
        optimizer.zero_grad()
        step = _global_step_offset(dataset_len, hps.n_batch_train, epoch, n_patch)
        kwargs = {
            'noisy1': image['noisy1'].to(hps.device),
            'noisy2': image['noisy2'].to(hps.device),
        }
        if 'nlf0' in image.keys():
            kwargs.update({'nlf0': image['nlf0'].to(hps.device), 'nlf1': image['nlf1'].to(hps.device)})
        if 'iso' in image.keys():
            kwargs.update({'iso': image['iso'].to(hps.device)})
        if 'cam' in image.keys():
            kwargs.update({'cam': image['cam'].to(hps.device)})

        loss, nll, mse = model.loss_u(**kwargs)
        losses.append(loss.item())
        nlls.append(nll)
        mses.append(mse)

        writer.add_scalar('Train Loss', losses[-1], step)
        writer.add_scalar('Train NLL per dim', nll, step)
        writer.add_scalar('Train MSE', mse, step)

        loss.backward()
        optimizer.step()

        if ((n_patch + 1) % report_interval == 0) or (n_patch + 1 == num_batches):
            logging.info(
                "Epoch %d | train batch %d/%d | loss=%.4f | nll=%.4f | mse=%.4f",
                epoch,
                n_patch + 1,
                num_batches,
                losses[-1],
                nll,
                mse,
            )

    elapsed = time.time() - start
    logging.info(
        "Epoch %d | train complete | mean_loss=%.4f | duration=%.1fs",
        epoch,
        float(np.mean(losses)) if losses else 0.0,
        elapsed,
    )
    return {
        'loss_mean': float(np.mean(losses)) if losses else 0.0,
        'nll_mean': float(np.mean(nlls)) if nlls else 0.0,
        'mse_mean': float(np.mean(mses)) if mses else 0.0,
        'time': elapsed,
    }


@torch.no_grad()
def validate_one_epoch(model: Noise2NoiseFlow, val_loader: DataLoader, hps, writer: SummaryWriter, epoch: int) -> Dict[str, float]:
    model.eval()
    losses, nlls, mses = [], [], []
    start = time.time()

    try:
        dataset_len = val_loader.dataset.__len__()
    except Exception:
        dataset_len = 0

    num_batches = len(val_loader)
    report_interval = max(1, num_batches // 4)
    logging.info("Epoch %d | val   | batches=%d", epoch, num_batches)

    for n_patch, image in enumerate(val_loader):
        step = _global_step_offset(dataset_len, hps.n_batch_test, epoch, n_patch)
        kwargs = {
            'noisy1': image['noisy1'].to(hps.device),
            'noisy2': image['noisy2'].to(hps.device),
        }
        if 'nlf0' in image.keys():
            kwargs.update({'nlf0': image['nlf0'].to(hps.device), 'nlf1': image['nlf1'].to(hps.device)})
        if 'iso' in image.keys():
            kwargs.update({'iso': image['iso'].to(hps.device)})
        if 'cam' in image.keys():
            kwargs.update({'cam': image['cam'].to(hps.device)})

        loss, nll, mse = model.loss_u(**kwargs)
        losses.append(loss.item())
        nlls.append(nll)
        mses.append(mse)

        writer.add_scalar('Validation Loss', losses[-1], step)
        writer.add_scalar('Validation NLL per dim', nll, step)
        writer.add_scalar('Validation MSE', mse, step)

        if ((n_patch + 1) % report_interval == 0) or (n_patch + 1 == num_batches):
            logging.info(
                "Epoch %d | val batch %d/%d | loss=%.4f | nll=%.4f | mse=%.4f",
                epoch,
                n_patch + 1,
                num_batches,
                losses[-1],
                nll,
                mse,
            )

    elapsed = time.time() - start
    logging.info(
        "Epoch %d | val complete   | mean_loss=%.4f | duration=%.1fs",
        epoch,
        float(np.mean(losses)) if losses else 0.0,
        elapsed,
    )
    return {
        'loss_mean': float(np.mean(losses)) if losses else 0.0,
        'nll_mean': float(np.mean(nlls)) if nlls else 0.0,
        'mse_mean': float(np.mean(mses)) if mses else 0.0,
        'time': elapsed,
    }


@torch.no_grad()
def test_one_epoch(model: Noise2NoiseFlow, test_loader: DataLoader, hps, writer: SummaryWriter, epoch: int) -> Dict[str, float]:
    nlls, mses, psnrs = [], [], []
    start = time.time()

    try:
        dataset_len = test_loader.dataset.__len__()
    except Exception:
        dataset_len = 0

    num_batches = len(test_loader)
    report_interval = max(1, num_batches // 4)
    logging.info("Epoch %d | test  | batches=%d", epoch, num_batches)

    for n_patch, image in enumerate(test_loader):
        step = _global_step_offset(dataset_len, hps.n_batch_test, epoch, n_patch)
        kwargs = {
            'x': image['noise'].to(hps.device),
            'clean': image['clean'].to(hps.device),
        }
        if 'nlf0' in image.keys():
            kwargs.update({'nlf0': image['nlf0'].to(hps.device), 'nlf1': image['nlf1'].to(hps.device)})
        if 'iso' in image.keys():
            kwargs.update({'iso': image['iso'].to(hps.device)})
        if 'cam' in image.keys():
            kwargs.update({'cam': image['cam'].to(hps.device)})

        nll, _ = model.loss_s(**kwargs)
        mse, psnr = model.mse_loss(kwargs['x'] + kwargs['clean'], kwargs['clean'])
        nlls.append(nll.item())
        mses.append(mse)
        psnrs.append(psnr)

        writer.add_scalar('Test NLL per dim', nlls[-1], step)
        writer.add_scalar('Test Denoiser MSE', mse, step)
        writer.add_scalar('Test Denoiser PSNR', psnr, step)

        if ((n_patch + 1) % report_interval == 0) or (n_patch + 1 == num_batches):
            logging.info(
                "Epoch %d | test batch %d/%d | nll=%.4f | mse=%.4f | psnr=%.2f",
                epoch,
                n_patch + 1,
                num_batches,
                nlls[-1],
                mse,
                psnr,
            )

    elapsed = time.time() - start
    logging.info(
        "Epoch %d | test complete  | mean_nll=%.4f | duration=%.1fs",
        epoch,
        float(np.mean(nlls)) if nlls else 0.0,
        elapsed,
    )
    return {
        'nll_mean': float(np.mean(nlls)) if nlls else 0.0,
        'mse_mean': float(np.mean(mses)) if mses else 0.0,
        'psnr_mean': float(np.mean(psnrs)) if psnrs else 0.0,
        'time': elapsed,
    }


@torch.no_grad()
def sample_epoch(model: Noise2NoiseFlow, test_loader: DataLoader, hps, writer: SummaryWriter, epoch: int, pat_stats: Dict) -> Dict[str, float]:
    """Sampling evaluation mirrors original script with optional fixed ISO/CAM path."""
    if not getattr(hps, 'do_sample', False):
        return {'loss_mean': 0.0, 'sdz_mean': 0.0, 'kldiv': [0.0, 0.0, 0.0, 0.0], 'time': 0.0}

    # Same defaults as original; sampling path retains iso/cam conditioning when available

    model.eval()
    start = time.time()

    sample_loss, sample_sdz = [], []
    n_models = 4 if getattr(hps, 'raw', False) else 3
    kldiv = np.zeros(n_models)
    count = 0

    num_batches = len(test_loader)
    report_interval = max(1, num_batches // 4)
    logging.info("Epoch %d | sample | batches=%d", epoch, num_batches)

    for n_patch, image in enumerate(test_loader):
        count += 1
        # step only used for TB visualization
        try:
            dataset_len = test_loader.dataset.__len__()
        except Exception:
            dataset_len = 0
        step = _global_step_offset(dataset_len, hps.n_batch_test, epoch, n_patch)

        kwargs = {
            'clean': image['clean'].to(hps.device),
            'eps_std': torch.tensor(hps.temp if hasattr(hps, 'temp') else 1.0, device=hps.device),
            'writer': writer,
            'step': step,
        }
        # Only attach NLF fields if present
        if 'nlf0' in image.keys() and 'nlf1' in image.keys():
            kwargs.update({
                'nlf0': image['nlf0'].to(hps.device),
                'nlf1': image['nlf1'].to(hps.device),
            })
        if 'iso' in image.keys():
            kwargs.update({'iso': image['iso'].to(hps.device)})
        if 'cam' in image.keys():
            kwargs.update({'cam': image['cam'].to(hps.device)})

        x_sample_val = model.sample(**kwargs)

        # evaluate NLL on samples
        eval_kwargs = {
            'x': x_sample_val,
            'clean': kwargs['clean'],
        }
        if kwargs.get('nlf0', None) is not None and kwargs.get('nlf1', None) is not None:
            eval_kwargs.update({'nlf0': kwargs['nlf0'], 'nlf1': kwargs['nlf1']})
        if kwargs.get('iso', None) is not None:
            eval_kwargs.update({'iso': kwargs['iso']})
        if kwargs.get('cam', None) is not None:
            eval_kwargs.update({'cam': kwargs['cam']})

        nll, sd_z = model.loss_s(**eval_kwargs)
        sample_loss.append(nll.item())
        sample_sdz.append(sd_z.item())

        # Marginal KL divergence
        vis_mbs_dir = os.path.join(hps.logdir, 'samples', f'samples_epoch_{epoch:04d}', f'samples_{float(getattr(hps, "temp", 1.0)):.1f}')
        sc_in_sd = pat_stats['sc_in_sd'] if isinstance(pat_stats, dict) else None
        if sc_in_sd is not None:
            kldiv_batch, cnt_batch = calc_kldiv_mb(
                image,
                x_sample_val.data.to('cpu'),
                vis_mbs_dir,
                sc_in_sd,
                n_models,
            )
            kldiv += kldiv_batch / cnt_batch

        if ((n_patch + 1) % report_interval == 0) or (n_patch + 1 == num_batches):
            logging.info(
                "Epoch %d | sample batch %d/%d | nll=%.4f | sd_z=%.4f",
                epoch,
                n_patch + 1,
                num_batches,
                sample_loss[-1],
                sample_sdz[-1],
            )

    elapsed = time.time() - start
    kldiv /= max(count, 1)
    kldiv_list = list(kldiv)

    logging.info(
        "Epoch %d | sample complete | mean_nll=%.4f | duration=%.1fs",
        epoch,
        float(np.mean(sample_loss)) if sample_loss else 0.0,
        elapsed,
    )

    return {
        'loss_mean': float(np.mean(sample_loss)) if sample_loss else 0.0,
        'sdz_mean': float(np.mean(sample_sdz)) if sample_sdz else 0.0,
        'kldiv': kldiv_list,
        'time': elapsed,
    }


# ------------------------------
# Orchestration
# ------------------------------
def run_training(hps) -> None:
    setup_logging()

    # Check/download data and setup basics
    hps.dataset_kind = 'custom'
    host = socket.gethostname()
    _ = host  # keep for parity; not used further

    setup_device_and_seed(hps)
    prepare_logdirs(hps)
    logging.debug('Data root = %s' % hps.sidd_path)
    logging.debug('Logging to ' + hps.logdir)
    if hps.dataset_kind == 'custom' and 'sdn' in (hps.arch or ''):
        logging.warning('Custom dataset detected but architecture includes "sdn" layers; ensure the model does not require ISO/CAM metadata.')

    # Data
    train_loader, val_loader, test_loader, x_shape = build_dataloaders(hps)
    pat_stats, nll_gauss, nll_sdn = compute_patch_stats_and_baselines(hps, test_loader, x_shape)

    # Model & optimizer
    model, optimizer = build_model_and_optimizer(hps)
    start_epoch = try_resume(hps, model, optimizer)

    # Loggers
    train_logger, validation_logger, test_logger, sample_logger, writer = create_loggers_and_writer(hps, start_epoch)

    test_nll_best = np.inf
    for epoch in range(start_epoch, hps.epochs):
        do_validation = is_validation_epoch(epoch, hps)
        is_best = 0
        logging.info("Epoch %d started | validation=%s", epoch, do_validation)

        # Train
        tr = train_one_epoch(model, optimizer, train_loader, hps, writer, epoch)
        writer.add_scalar('Train Epoch Loss', tr['loss_mean'], epoch)
        writer.add_scalar('Train Epoch NLL per dim', tr['nll_mean'], epoch)
        writer.add_scalar('Train Epoch MSE', tr['mse_mean'], epoch)

        train_logger.log({
            'epoch': epoch,
            'train_time': int(tr['time']),
            'NLL': tr['loss_mean'],
            'NLL_G': nll_gauss,
            'NLL_SDN': nll_sdn,
        })

        if do_validation:
            # Validation
            val = validate_one_epoch(model, val_loader, hps, writer, epoch)
            writer.add_scalar('Validation Epoch Loss', val['loss_mean'], epoch)
            writer.add_scalar('Validation Epoch NLL per dim', val['nll_mean'], epoch)
            writer.add_scalar('Validation Epoch MSE', val['mse_mean'], epoch)

            validation_logger.log({
                'epoch': epoch,
                'NLL': val['loss_mean'],
                'NLL_G': nll_gauss,
                'NLL_SDN': nll_sdn,
            })

            # Test
            ts = test_one_epoch(model, test_loader, hps, writer, epoch)
            writer.add_scalar('Test Epoch NLL per dim', ts['nll_mean'], epoch)
            writer.add_scalar('Test Epoch Denoiser MSE', ts['mse_mean'], epoch)
            writer.add_scalar('Test Epoch Denoiser PSNR', ts['psnr_mean'], epoch)

            # Save checkpoints
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(hps.model_save_dir, f'epoch_{epoch}_nf_model_net.pth'),
            )

            if ts['nll_mean'] < test_nll_best:
                test_nll_best = ts['nll_mean']
                save_checkpoint(model, optimizer, epoch, os.path.join(hps.model_save_dir, 'best_model.pth'))
                is_best = 1

            # Persist per-epoch evaluation metrics alongside the current best-model flag
            test_logger.log({
                'epoch': epoch,
                'NLL': ts['nll_mean'],
                'NLL_G': nll_gauss,
                'NLL_SDN': nll_sdn,
                'msg': is_best,
            })

            # Sampling
            hps.temp = 1.0
            sm = sample_epoch(model, test_loader, hps, writer, epoch, pat_stats)
            if sm['kldiv']:
                # Align keys with original logger expectations
                kldiv_list = sm['kldiv']
                # When raw, we expect 4 models: [G, NLF, NF, R]; otherwise 3
                kld_g = kldiv_list[0] if len(kldiv_list) >= 1 else 0
                kld_nlf = kldiv_list[1] if len(kldiv_list) == 4 else 0
                kld_nf = kldiv_list[1] if len(kldiv_list) >= 2 else 0
                kld_r = kldiv_list[2] if len(kldiv_list) >= 3 else 0
            else:
                kld_g = kld_nf = kld_r = kld_nlf = 0

            sample_logger.log({
                'epoch': epoch,
                'NLL': sm['loss_mean'],
                'NLL_G': nll_gauss,
                'NLL_SDN': nll_sdn,
                'sdz': sm['sdz_mean'],
                'sample_time': sm['time'],
                'KLD_NLF': kld_nlf,
                'KLD_G': kld_g,
                'KLD_NF': kld_nf,
                'KLD_R': kld_r,
            })

            writer.add_scalar('Sample Epoch KLD', kld_nf, epoch)
            writer.add_scalar('Sample Epoch NLL per dim', sm['loss_mean'], epoch)

            # Console summary (kept close to original fields)
            print(
                "{}, epoch: {}, tr_loss: {:.3f}, val_loss : {:.3f} ts_loss: {:.3f}, ts_psnr: {:.2f}, sm_loss: {:.3f}, sm_kld: {:.4f}, tr_time: {:d}, val_time: {:d}, ts_time: {:d}, sm_time: {:d}, T_time: {:d}, best:{}".format(
                    hps.logdirname,
                    epoch,
                    tr['loss_mean'],
                    val['loss_mean'],
                    ts['nll_mean'],
                    ts['psnr_mean'],
                    sm['loss_mean'],
                    kld_nf,
                    int(tr['time']),
                    int(val['time']),
                    int(ts['time']),
                    int(sm['time']),
                    int(tr['time'] + ts['time'] + sm['time']),
                    is_best,
                )
            )

            logging.info(
                "Epoch %d summary | train=%.4f | val=%.4f | test=%.4f | smpl=%.4f | best=%s",
                epoch,
                tr['loss_mean'],
                val['loss_mean'],
                ts['nll_mean'],
                sm['loss_mean'],
                bool(is_best),
            )
        else:
            logging.info(
                "Epoch %d summary | train=%.4f (validation skipped)",
                epoch,
                tr['loss_mean'],
            )

    writer.close()


def main(hps):
    total_time = time.time()
    run_training(hps)
    total_time = time.time() - total_time
    logging.info('Training finished in %.1fs', total_time)
    logging.debug("Finished!")


if __name__ == "__main__":
    hps = arg_parser()
    main(hps)
