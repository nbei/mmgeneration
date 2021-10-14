import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.parallel.utils import is_module_wrapper

from mmgen.models.builder import MODULES, build_module
from ..stylegan.generator_discriminator_v2 import StyleGAN2Discriminator
from .augment import AugmentPipe
from .misc import constant


@MODULES.register_module()
class ADAAug(nn.Module):

    def __init__(self,
                 aug_pipeline=None,
                 update_interval=4,
                 augment_initial_p=0.,
                 ada_target=0.6,
                 ada_kimg=500,
                 use_slow_aug=False):
        super().__init__()
        self.aug_pipeline = AugmentPipe(**aug_pipeline)
        self.update_interval = update_interval
        self.ada_kimg = ada_kimg
        self.ada_target = ada_target

        self.aug_pipeline.p.copy_(torch.tensor(augment_initial_p))

        # this log buffer stores two numbers: num_scalars, sum_scalars.
        self.register_buffer('log_buffer', torch.zeros((2, )))

    def update(self, iteration=0, num_batches=0):

        if (iteration + 1) % self.update_interval == 0:

            adjust_step = float(num_batches * self.update_interval) / float(
                self.ada_kimg * 1000.)

            # get the mean value as the ada heuristic
            ada_heuristic = self.log_buffer[1] / self.log_buffer[0]
            adjust = np.sign(ada_heuristic.item() -
                             self.ada_target) * adjust_step
            # update the augment p
            # Note that p may be bigger than 1.0
            self.aug_pipeline.p.copy_((self.aug_pipeline.p + adjust).max(
                constant(0, device=self.log_buffer.device)))

            self.log_buffer = self.log_buffer * 0.


@MODULES.register_module()
class ADAStyleGAN2Disc(StyleGAN2Discriminator):

    def __init__(self, *args, data_aug=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_ada = data_aug is not None

        if self.with_ada:
            self.ada_aug = build_module(data_aug)

    def forward(self, x):
        if self.with_ada:
            x = self.ada_aug.aug_pipeline(x)
        return super().forward(x)


def ada_update_function(disc, real_logit, iteration=0, fixed_p=False):
    module = disc if not is_module_wrapper(disc) else disc.module
    ada_module = module.ada_aug

    num_batches = real_logit.size(0)

    if real_logit.ndim == 1:
        pass
    elif real_logit.ndim == 2 and real_logit.shape[1] > 1:
        real_logit = real_logit.mean(dim=1)

    real_logit_sign = real_logit.sign().sum()
    dist.all_reduce(real_logit_sign)
    total_batches = dist.get_world_size() * num_batches
    ada_heuristic = real_logit_sign.item() / float(total_batches)

    if not fixed_p:
        # update num scalars
        ada_module.log_buffer[0] = ada_module.log_buffer[0] + float(
            total_batches)
        # update sum of scalars
        # This implementaion will cause RAM leak.
        # ada_module.log_buffer[1] = ada_module.log_buffer[1] + real_logit_sign
        ada_module.log_buffer[1] += real_logit_sign.item()

        ada_module.update(iteration=iteration, num_batches=total_batches)

    output_dict = dict(
        heuristic=torch.tensor(ada_heuristic).to(real_logit),
        p=ada_module.aug_pipeline.p)

    return output_dict


@MODULES.register_module()
class AdaUpdater(nn.Module):

    def __init__(self, data_info=None, fixed_p=False, loss_name='ada'):
        super().__init__()
        self._loss_name = loss_name
        self.data_info = data_info
        self.fixed_p = fixed_p

    def forward(self, *args, **kwargs):
        # use data_info to build computational path
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')

            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            kwargs.update(dict(fixed_p=self.fixed_p))
            return ada_update_function(**kwargs)
        else:
            return ada_update_function(*args, fixed_p=self.fixed_p, **kwargs)

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
