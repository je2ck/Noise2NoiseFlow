import torch
from torch import nn
import numpy as np

class Gain(nn.Module):
    def __init__(self, name='gain', device='cuda'):
        super(Gain, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=True)
        self.name = name

    def _inverse(self, z, **kwargs):
        x = z * self.scale
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        scale = self.scale + (x * 0.0)

        if 'writer' in kwargs.keys():
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_mean', torch.mean(scale), kwargs['step'])
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_min', torch.min(scale), kwargs['step'])
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_max', torch.max(scale), kwargs['step'])

        z = x / scale
        log_abs_det_J_inv = - torch.sum(torch.log(scale), dim=[1, 2, 3])
        return z, log_abs_det_J_inv

class GainExp2(nn.Module):
    def __init__(self, gain_scale, param_inits, device='cpu', name='gain'):
        super(GainExp2, self).__init__()
        self.name = name
        self._gain_scale = gain_scale(param_inits, device=device, name='gain_layer_gain_scale')

    def _inverse(self, z, **kwargs):
        scale, _ = self._gain_scale(kwargs['clean'], kwargs['iso'], kwargs['cam'])
        x = z * scale
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        writer = kwargs['writer'] if 'writer' in kwargs.keys() else None
        step = kwargs['step'] if 'step' in kwargs.keys() else None
    
        scale, _ = self._gain_scale(kwargs['clean'], kwargs['iso'], kwargs['cam'], writer, step)

        if writer:
            writer.add_scalar('model/' + self.name + '_scale_mean', torch.mean(scale), step)

        z = x / scale

        log_abs_det_J_inv = - torch.sum(torch.log(scale), dim=[1, 2, 3])

        return z, log_abs_det_J_inv
