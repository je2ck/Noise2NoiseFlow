import torch
from torch import nn
import torch.nn.functional as F

class Logit(nn.Module):
    def __init__(self, temperature=1, eps=1e-6, device='cpu', name='logit'):
        super(Logit, self).__init__()
        self.name = name
        self.eps = eps
        self.register_buffer('temperature', torch.tensor([temperature], device=device))

    def _inverse(self, z, **kwargs):
        z = self.temperature * z
        x = torch.sigmoid(z)
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        """
        ldj explanation:
        log(x/(1-x)) = z => x/(1-x) = exp(z) (property 1) => (1-x)/x = 1/exp(x) => 1/x - 1 = 1/exp(z)
        => 1/x = 1 + 1/exp(z) (property 2)

        softplus(-z) + softplus(z) = log(1 + exp(-z) + log(1 + exp(z)))
                                   = log(1 + 1/exp(z)) + log(1 + exp(z))
                                   = log(1/x) + log(1 + x/(1-x))
                                   = -log(x) + log((1-x+x)/(1-x))
                                   = -log(x) + log(1/(1-x))
                                   = -log(x) - log(1-x)
        """
        z = (1 / self.temperature) * (torch.logit(x, eps=self.eps))
        ldj = torch.sum( - (torch.log(self.temperature) - F.softplus(-self.temperature * z) - F.softplus(self.temperature * z)), dim=[1, 2, 3])
        return z, ldj
