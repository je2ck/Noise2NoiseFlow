import torch
from torch import nn
import numpy as np
from functools import partial

from model.flow_layers.conv2d1x1 import Conv2d1x1
from model.flow_layers.affine_coupling import AffineCoupling, ShiftAndLogScale, ConditionalAffine
from model.flow_layers.signal_dependant import SignalDependant
from model.flow_layers.gain import Gain
from model.flow_layers.utils import SdnModelScale
from model.flow_layers.basden import BasdenFlowLayer, BasdenAdaptor
from model.flow_layers.squeeze import SqueezeLayer, UnsqueezeLayer
# from model.flow_layers.linear_transformation import LinearTransformation

class NoiseFlow(nn.Module):

    def __init__(self, x_shape, arch, flow_permutation, param_inits, basden_config=None, lu_decomp=None, device='cuda', *_, **__):
        super(NoiseFlow, self).__init__()
        self.arch = arch
        self.flow_permutation = flow_permutation
        self.param_inits = param_inits
        self.decomp = lu_decomp
        self.device = device
        self.basden_config = basden_config
        self.model = nn.ModuleList(self.noise_flow_arch(x_shape))

    def noise_flow_arch(self, x_shape):
        """
        Tokens:
          basden  - physical EMCCD model (BasdenAdaptor + BasdenFlowLayer)
          cond    - ConditionalAffine(only_clean=True): per-pixel shift/log_scale
                    predicted from the denoised (clean) image.
                    Targets signal-dependent residual that Basden cannot capture
                    (panel 7 of residual_diag).
          sq      - Squeeze (space-to-depth, factor 2): [C,H,W] -> [4C, H/2, W/2]
          usq     - Unsqueeze (depth-to-space, factor 2): inverse of sq
          unc     - Conv2d1x1 permutation + AffineCoupling
          sdn     - SignalDependant (needs clean/iso/cam)
          gain    - scalar gain
        Example:  basden|cond              (current recommendation)
                  basden|sq|unc|unc|unc|usq (if spatial structure remains)
        """
        arch_lyrs = self.arch.split('|')
        bijectors = []
        cur = list(x_shape)   # [C, H, W] — updated by sq/usq
        for i, lyr in enumerate(arch_lyrs):
            if lyr == 'basden':
                print('|-BasdenAdaptor')
                bijectors.append(
                    BasdenAdaptor(num_channels=cur[0], device=self.device)
                )
                print('|-BasdenFlowLayer')
                bijectors.append(
                    BasdenFlowLayer(config=self.basden_config, device=self.device)
                )

            elif lyr == 'sq':
                print('|-Squeeze   in={}'.format(cur), end='')
                bijectors.append(SqueezeLayer(factor=2, level=i, name='sq_%d' % i))
                cur = [cur[0] * 4, cur[1] // 2, cur[2] // 2]
                print(' -> out={}'.format(cur))

            elif lyr == 'usq':
                print('|-Unsqueeze in={}'.format(cur), end='')
                bijectors.append(UnsqueezeLayer(factor=2, level=i, name='usq_%d' % i))
                cur = [cur[0] // 4, cur[1] * 2, cur[2] * 2]
                print(' -> out={}'.format(cur))

            elif lyr == 'unc':
                if self.flow_permutation == 0:
                    pass
                elif self.flow_permutation == 1:
                    print('|-Conv2d1x1 (C={})'.format(cur[0]))
                    bijectors.append(
                        Conv2d1x1(
                            num_channels=cur[0],
                            LU_decomposed=self.decomp,
                            name='Conv2d_1x1_{}'.format(i)
                        )
                    )
                else:
                    print('|-No permutation specified. Not using any.')

                print('|-AffineCoupling (C={})'.format(cur[0]))
                bijectors.append(
                    AffineCoupling(
                        x_shape=tuple(cur),
                        shift_and_log_scale=ShiftAndLogScale,
                        name='unc_%d' % i,
                        device=self.device
                    )
                )

            elif lyr == 'sdn':
                print('|-SignalDependant')
                bijectors.append(
                    SignalDependant(
                        name='sdn_%d' % i,
                        scale=SdnModelScale,
                        param_inits=self.param_inits
                    )
                )

            elif lyr == 'gain':
                print('|-Gain')
                bijectors.append(Gain(name='gain_%d' % i, device=self.device))

            elif lyr == 'cond':
                # ShiftAndLogScale with FIXED, non-trainable scale cap.
                # Bound MUST be wide enough to allow meaningful per-clean
                # log_scale variation (Panel 7 shows var(z|c) up to 8, so
                # cond needs log_scale ≈ -log(sqrt(8)) = -1.04). A tight
                # cap (e.g., 0.5) causes cond to saturate and give up
                # per-c learning (training pathology observed with 0.5).
                # 1.5 allows log_scale ∈ (-1.5, +1.5), i.e. scale ∈ (0.22, 4.48)
                # — enough to correct Panel 7 while still preventing full
                # mode collapse observed with unbounded cap.
                bounded_sls = partial(
                    ShiftAndLogScale,
                    scale_init=1.5,
                    scale_trainable=False,
                )
                print('|-ConditionalAffine(only_clean=True, C={}, '
                      'scale∈(-1.5,1.5))'.format(cur[0]))
                bijectors.append(
                    ConditionalAffine(
                        x_shape=tuple(cur),
                        shift_and_log_scale=bounded_sls,
                        encoder=None,            # only_clean=True 에선 사용 안 함
                        name='cond_%d' % i,
                        device=self.device,
                        only_clean=True,
                    )
                )

            else:
                raise ValueError("Unknown arch token: {!r}".format(lyr))

        if cur != list(x_shape):
            raise ValueError(
                "Arch must return to original shape {} but ended at {}. "
                "Each 'sq' needs a matching 'usq'.".format(list(x_shape), cur)
            )

        return bijectors

    def forward(self, x, **kwargs):
        z = x
        objective = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
        for bijector in self.model:
            z, log_abs_det_J_inv = bijector._forward_and_log_det_jacobian(z, **kwargs)
            objective += log_abs_det_J_inv

            if 'writer' in kwargs.keys():
                kwargs['writer'].add_scalar('model/' + bijector.name, torch.mean(log_abs_det_J_inv), kwargs['step'])
        return z, objective

    def _loss(self, x, **kwargs):
        z, objective = self.forward(x, **kwargs)
        # base measure
        logp, _ = self.prior("prior", x)

        log_z = logp(z)
        objective += log_z

        if 'writer' in kwargs.keys():
            kwargs['writer'].add_scalar('model/log_z', torch.mean(log_z), kwargs['step'])
            kwargs['writer'].add_scalar('model/z', torch.mean(z), kwargs['step'])
        nobj = - objective
        # std. dev. of z
        mu_z = torch.mean(x, dim=[1, 2, 3])
        var_z = torch.var(x, dim=[1, 2, 3])
        sd_z = torch.mean(torch.sqrt(var_z))

        return nobj, sd_z

    def loss(self, x, **kwargs):
        batch_average = torch.mean(x, dim=0)
        # if 'writer' in kwargs.keys():
        #     kwargs['writer'].add_histogram('real_noise', batch_average, kwargs['step'])
        #     kwargs['writer'].add_scalar('real_noise_std', torch.std(batch_average), kwargs['step'])

        nll, sd_z = self._loss(x=x, **kwargs)
        nll_dim = torch.mean(nll) / np.prod(x.shape[1:])
        # nll_dim = torch.mean(nll)      # The above line should be uncommented

        return nll_dim, sd_z

    def inverse(self, z, **kwargs):
        x = z
        for bijector in reversed(self.model):
            x = bijector._inverse(x, **kwargs)
        return x
    
    def sample(self, eps_std=None, **kwargs):
        _, sample = self.prior("prior", kwargs['clean'])
        z = sample(eps_std)
        x = self.inverse(z, **kwargs)
        batch_average = torch.mean(x, dim=0)
        if 'writer' in kwargs.keys():
            kwargs['writer'].add_histogram('sample_noise', batch_average, kwargs['step'])
            kwargs['writer'].add_scalar('sample_noise_std', torch.std(batch_average), kwargs['step'])

        return x

    def prior(self, name, x):
        n_z = x.shape[1]
        h = torch.zeros([x.shape[0]] +  [2 * n_z] + list(x.shape[2:4]), device=x.device)
        pz = gaussian_diag(h[:, :n_z, :, :], h[:, n_z:, :, :])

        def logp(z1):
            objective = pz.logp(z1)
            return objective

        def sample(eps_std=None):
            if eps_std is not None:
                z = pz.sample2(pz.eps * torch.reshape(eps_std, [-1, 1, 1, 1]))
            else:
                z = pz.sample
            return z

        return logp, sample

def gaussian_diag(mean, logsd):
    class o(object):
        pass

    o.mean = mean
    o.logsd = logsd
    o.eps = torch.normal(torch.zeros(mean.shape, device=mean.device), torch.ones(mean.shape, device=mean.device))
    o.sample = mean + torch.exp(logsd) * o.eps
    o.sample2 = lambda eps: mean + torch.exp(logsd) * eps

    o.logps = lambda x: -0.5 * (np.log(2 * np.pi) + 2. * o.logsd + (x - o.mean) ** 2 / torch.exp(2. * o.logsd))
    o.logp = lambda x: torch.sum(o.logps(x), dim=[1, 2, 3])
    o.get_eps = lambda x: (x - mean) / torch.exp(logsd)
    return o
