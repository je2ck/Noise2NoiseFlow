import torch
from torch import nn

from model.noise_flow import NoiseFlow
from model.dncnn import DnCNN
from utils.train_utils import weights_init_kaiming, weights_init_orthogonal
from model.unet import UNet
from model.flow_layers.poisson_prior import PoissonPrior
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

class Noise2NoiseFlow(nn.Module):
    def __init__(self, x_shape, arch, flow_permutation, param_inits, lu_decomp, basden_config,
                 denoiser_model='dncnn', dncnn_num_layers=9, lmbda=262144, device='cuda',
                 prior_cfg=None):
        super(Noise2NoiseFlow, self).__init__()

        self.noise_flow = NoiseFlow(x_shape, arch, flow_permutation, param_inits, basden_config, lu_decomp, device=device)
        if denoiser_model == 'dncnn':
            self.denoiser = DnCNN(x_shape[0], dncnn_num_layers)
            # TODO: self.dncnn should be named self.denoiser by definition, but I changed it here since i needed it to be backward compatible for loading previous models for sampling.
            # self.denoiser.apply(weights_init_kaiming)
            self.denoiser.apply(weights_init_orthogonal)
        elif denoiser_model == 'unet':
            self.denoiser = UNet(in_channels=4, out_channels=4)

        self.denoiser_loss = nn.MSELoss(reduction='mean')
        self.lmbda = lmbda

        # Optional Poisson prior on x_hat (atom-only mask).
        # prior_cfg dict keys:
        #   'enabled' (bool), 'lmbda_prior' (float), 'lambda_atom' (float),
        #   'atom_threshold_photon' (float), 'learnable' (bool),
        #   'vmin', 'vmax' (float — normalization range for photon conversion),
        #   'sensitivity', 'em_gain' (float — ADU -> photon).
        # If None or enabled=False, prior is disabled and existing loss is unchanged.
        self.prior_cfg = prior_cfg
        self.prior_flow = None
        if prior_cfg is not None and prior_cfg.get('enabled', False):
            self.prior_flow = PoissonPrior(
                lambda_init=prior_cfg['lambda_atom'],
                sum_threshold_photon=prior_cfg.get('sum_threshold_photon', 2.0),
                roi_size=prior_cfg.get('roi_size', 5),
                learnable=prior_cfg.get('learnable', False),
            )
            self.lmbda_prior = float(prior_cfg.get('lmbda_prior', 0.0))
            self._photon_scale = float(
                (prior_cfg['vmax'] - prior_cfg['vmin'])
                * prior_cfg['sensitivity'] / prior_cfg['em_gain']
            )
        else:
            self.lmbda_prior = 0.0
            self._photon_scale = 1.0

    def _to_photon(self, x_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized [0,1] x_hat to photon units (bg ~ vmin assumed)."""
        return x_normalized * self._photon_scale

    def _prior_nll(self, *denoised_tensors) -> torch.Tensor:
        """Compute prior NLL averaged across provided denoised tensors.
        Returns scalar tensor. Safe to call when prior is disabled (returns 0)."""
        if self.prior_flow is None or self.lmbda_prior <= 0.0:
            # Return a tensor on the same device, differentiable no-op
            ref = denoised_tensors[0]
            return ref.new_zeros(())
        # Use only first channel (others are aux / duplicated) for photon interpretation
        nll_vals = []
        for d in denoised_tensors:
            d_photon = self._to_photon(d[:, :1])
            nll_vals.append(self.prior_flow(d_photon))
        return sum(nll_vals) / len(nll_vals)

    def denoise(self, noisy, clip=True):
        denoised = self.denoiser(noisy)
        if clip:
            denoised = torch.clamp(denoised, 0., 1.)

        return denoised

    def forward_u(self, noisy, **kwargs):
        denoised = self.denoise(noisy)
        kwargs.update({'clean' : denoised})
        noise = noisy - denoised

        z, objective = self.noise_flow.forward(noise, **kwargs)

        return z, objective, denoised

    def symmetric_loss(self, noisy1, noisy2, **kwargs):
        denoised1 = self.denoise(noisy1)
        denoised2 = self.denoise(noisy2)
        
        noise1 = noisy1 - denoised2
        noise2 = noisy2 - denoised1

        kwargs.update({'clean' : denoised2})
        nll1, _ = self.noise_flow.loss(noise1, **kwargs)

        kwargs.update({'clean' : denoised1})
        nll2, _ = self.noise_flow.loss(noise2, **kwargs)

        nll = (nll1 + nll2) / 2
        return nll

    def symmetric_loss_with_mse(self, noisy1, noisy2, **kwargs):
        denoised1_raw = self.denoise(noisy1, clip=False)
        denoised2_raw = self.denoise(noisy2, clip=False)

        mse_loss1 = self.denoiser_loss(denoised1_raw, noisy2)
        mse_loss2 = self.denoiser_loss(denoised2_raw, noisy1)

        # Prior NLL on UN-clipped denoised output so saturated atoms can
        # freely reach λ_atom. Gradient unaffected by the later clamp
        # (clamp is a separate copy used for the residual flow branch).
        prior_nll = self._prior_nll(denoised1_raw, denoised2_raw)

        denoised1 = torch.clamp(denoised1_raw, 0., 1.)
        denoised2 = torch.clamp(denoised2_raw, 0., 1.)

        noise1 = noisy1 - denoised2
        noise2 = noisy2 - denoised1

        kwargs.update({'clean' : denoised2})
        nll1, _ = self.noise_flow.loss(noise1, **kwargs)

        kwargs.update({'clean' : denoised1})
        nll2, _ = self.noise_flow.loss(noise2, **kwargs)

        nll = (nll1 + nll2) / 2
        mse_loss = (mse_loss1 + mse_loss2) / 2

        return nll, mse_loss, prior_nll



    def _loss_u(self, noisy1, noisy2, **kwargs):
        denoised1 = self.denoise(noisy1, clip=False)

        mse_loss = self.denoiser_loss(denoised1, noisy2)

        denoised1 = torch.clamp(denoised1, 0., 1.)

        noise = noisy1 - denoised1
        kwargs.update({'clean' : denoised1})
        nll, _ = self.noise_flow.loss(noise, **kwargs)

        return nll, mse_loss

    def loss_u(self, noisy1, noisy2, **kwargs):
        # return self.symmetric_loss(noisy1, noisy2, **kwargs), 0, 0

        # nll, mse = self._loss_u(noisy1, noisy2, **kwargs)
        nll, mse, prior_nll = self.symmetric_loss_with_mse(noisy1, noisy2, **kwargs)

        total = nll + self.lmbda * mse
        if self.lmbda_prior > 0.0:
            total = total + self.lmbda_prior * prior_nll
        return total, nll.item(), mse.item(), float(prior_nll.item())
        # return nll, nll.item(), mse.item()

    def forward_s(self, noise, **kwargs):
        return self.noise_flow.forward(noise, **kwargs)

    def _loss_s(self, x, **kwargs):
        return self.noise_flow._loss(x, **kwargs)

    def loss_s(self, x, **kwargs):
        return self.noise_flow.loss(x, **kwargs)

    def mse_loss(self, noisy, clean, **kwargs):
        denoised = self.denoise(noisy, clip=False)
        mse_loss = self.denoiser_loss(denoised, clean)
        psnr = batch_PSNR(denoised, clean, 1.)
        return mse_loss.item(), psnr

    def sample(self, eps_std=None, **kwargs):
        return self.noise_flow.sample(eps_std, **kwargs)
