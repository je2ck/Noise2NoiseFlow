"""
ROI-sum Poisson prior on DnCNN output.

Design:
  At each pixel, compute the sum of x_hat over a (roi_size x roi_size) window
  centered there (sliding-window sum via box-filter convolution). Detect
  'atom candidates' as local maxima of this sum-map exceeding a threshold.
  Apply a continuous Poisson NLL on the detected ROI sums with rate λ_atom
  (= expected TOTAL photon count from one atom, i.e. the `<photon>` value
  reported by make_fidelity_table.py).

Why ROI-sum instead of per-pixel:
  The per-pixel rate is dominated by PSF shape — peak ~2-4 ph, halo ~0.1-0.8 ph.
  A single per-pixel λ can't fit both. Integrating over the 5x5 ROI removes
  this dependence: the TOTAL photon count ≈ atom emission rate, directly
  modelable by Poisson(λ_atom).

The selection (local-max + threshold) is non-differentiable by design —
gradients flow only through the sum operation into x_hat, which is correct.

NLL:  -log p(S; λ) = λ + log Γ(S + 1) - S · log λ,  S = ROI sum
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoissonPrior(nn.Module):
    """Prior on detected atom ROI sums. Two modes:
        - 'poisson_nll': continuous Poisson NLL
                         ℓ = λ + log Γ(S+1) - S log λ
        - 'l2':          squared-error regression
                         ℓ = (S - λ)^2

    L2 provides much stronger gradient far from λ (linear in |S-λ|)
    vs Poisson (digamma-bounded, gets weaker as S → 0).

    Args:
        lambda_init: expected ROI-sum photon count for an atom (e.g. 6.4 @ 5ms).
        sum_threshold_photon: minimum ROI sum for a position to be counted
            as an atom candidate. Keep ≥ 2 ph to reject background noise.
        roi_size: window side length (default 5; matches make_fidelity_table DX/DY=2).
        learnable: if True, λ is a free parameter.
        mode: 'poisson_nll' (default) or 'l2'.
    """

    def __init__(self, lambda_init: float,
                 sum_threshold_photon: float = 2.0,
                 roi_size: int = 5,
                 learnable: bool = False,
                 mode: str = 'poisson_nll'):
        super().__init__()
        log_lam = math.log(max(float(lambda_init), 1e-3))
        if learnable:
            self.log_lambda = nn.Parameter(torch.tensor(log_lam, dtype=torch.float32))
        else:
            self.register_buffer('log_lambda',
                                 torch.tensor(log_lam, dtype=torch.float32))
        self.sum_threshold_photon = float(sum_threshold_photon)
        self.roi_size = int(roi_size)
        assert self.roi_size % 2 == 1, "roi_size must be odd"
        self.half = self.roi_size // 2
        assert mode in ('poisson_nll', 'l2'), f"Unknown prior mode: {mode}"
        self.mode = mode
        # Box-filter kernel for sliding 5x5 sum (per-channel, in/out = 1).
        self.register_buffer(
            'box_kernel',
            torch.ones(1, 1, self.roi_size, self.roi_size, dtype=torch.float32),
        )

    @property
    def lam(self) -> torch.Tensor:
        return torch.exp(self.log_lambda)

    def forward(self, x_hat_photon: torch.Tensor) -> torch.Tensor:
        """
        x_hat_photon: (B, 1, H, W) in photon units. Must already be bg-subtracted
                      (vmin ≈ background in the upstream normalization).
        Returns mean prior loss over all detected atom ROIs in the batch.
        If no ROI is detected, returns 0.
        """
        # 1. Sliding ROI sum via conv: roi_sum[b,0,y,x] = Σ_{5x5 around (y,x)} x_hat
        roi_sum = F.conv2d(x_hat_photon, self.box_kernel, padding=self.half)

        # 2. Non-max suppression: keep positions where roi_sum is a local max
        #    within a roi_size neighborhood.
        pooled = F.max_pool2d(roi_sum,
                              kernel_size=self.roi_size,
                              stride=1,
                              padding=self.half)
        is_local_max = (roi_sum == pooled)

        # 3. Threshold on the ROI sum itself (reject faint / noise-only peaks).
        is_atom = is_local_max & (roi_sum > self.sum_threshold_photon)

        n_atom = is_atom.sum()
        if n_atom.item() == 0:
            return x_hat_photon.new_zeros(())

        atom_sums = roi_sum[is_atom]
        lam = self.lam

        if self.mode == 'l2':
            loss = (atom_sums - lam) ** 2
        else:  # poisson_nll
            atom_sums = torch.clamp(atom_sums, min=1e-3)
            loss = lam + torch.lgamma(atom_sums + 1.0) - atom_sums * torch.log(lam)
        return loss.mean()
