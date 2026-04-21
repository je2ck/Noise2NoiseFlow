"""
Exact continuous Poisson prior on DnCNN output.

Not a flow layer in the invertible sense — just a negative-log-likelihood
evaluator plugged into the training loss. Change-of-variables would give
an equivalent Poisson-CDF flow, but differentiating gammaincc w.r.t. its
first argument is nontrivial in PyTorch; the density form below is
mathematically equivalent for gradient purposes and trivial to implement.

Ilienko continuous Poisson density:
    p(x; lam) = lam^x * exp(-lam) / Gamma(x + 1),   x >= 0
    -log p     = lam + log Gamma(x + 1) - x log lam
"""

import math
import torch
import torch.nn as nn


class PoissonPrior(nn.Module):
    """Negative log-likelihood of x_hat under continuous Poisson(lam).

    Operates in photon units. The caller converts normalized / ADU
    outputs to photons before calling forward().
    """

    def __init__(self, lambda_init: float, learnable: bool = False,
                 atom_threshold_photon: float = 1.5):
        super().__init__()
        log_lam = math.log(max(float(lambda_init), 1e-3))
        if learnable:
            self.log_lambda = nn.Parameter(torch.tensor(log_lam, dtype=torch.float32))
        else:
            self.register_buffer('log_lambda',
                                 torch.tensor(log_lam, dtype=torch.float32))
        # Only apply prior where x_hat > threshold (photon units).
        # Captures "atom-like" pixels and skips background.
        self.atom_threshold_photon = float(atom_threshold_photon)

    @property
    def lam(self) -> torch.Tensor:
        return torch.exp(self.log_lambda)

    def forward(self, x_hat_photon: torch.Tensor) -> torch.Tensor:
        """
        x_hat_photon: (B, C, H, W) in photon units (already bg-subtracted
                      and bias-free).
        Returns scalar mean NLL over atom-mask pixels.
        If no atom-mask pixels in batch, returns 0.
        """
        lam = self.lam
        # Mask atom-like pixels dynamically from the denoiser output itself.
        # Self-consistent: the denoiser declares what's an atom; prior calibrates it.
        atom_mask = x_hat_photon > self.atom_threshold_photon
        n_atom = atom_mask.sum()
        if n_atom.item() == 0:
            return x_hat_photon.new_zeros(())

        x_atom = x_hat_photon[atom_mask]
        # Guard against accidental negatives leaking through (clip=False path).
        x_atom = torch.clamp(x_atom, min=0.0)
        # -log p(x; lam) = lam + log Gamma(x + 1) - x log lam
        nll = lam + torch.lgamma(x_atom + 1.0) - x_atom * torch.log(lam)
        return nll.mean()
