# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils.

"""


import os
import time
import psutil
import torch
import torchvision.transforms as transforms
from .ddp_utils import get_ddp_rank


process = psutil.Process(os.getpid())  # for reporting RAM usage


def log(s, *args, **kwargs):
    if get_ddp_rank() == 0:
        print(s, *args, **kwargs)


class Logger:
    def __init__(self, max_steps, start_step=0):
        self.max_steps = max_steps
        self.start_step = start_step
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            torch.cuda.synchronize()
        self.start_time = time.time()

    def __call__(self, step, metrics_dict):

        if self.is_cuda:
            torch.cuda.synchronize()
        log('batch={:04d}/{:04d} \t {} \t {:.4f} (sec/batch), mem ram/gpu: {:.2f}/{} (G)'.
            format(step, self.max_steps,
                   '\t'.join(['{}={:.4f}'.format(m, v) for m, v in metrics_dict.items()]),
                   (time.time() - self.start_time) / max(1, (step + 1 - self.start_step)),
                   process.memory_info().rss / 10 ** 9,
                   ('%.2f' % (torch.cuda.memory_reserved(0) / 10 ** 9)) if self.is_cuda else 'nan'),
            flush=True)


def print_grads(model, verbose=True):

    grads_table, norms_table = {}, {}

    if verbose:
        print('\n ======== gradient and param norms (sorted by grads) ========')
    norm_type = 2
    grads, norms, shapes = {}, {}, {}
    for i, (n, p) in enumerate(model.named_parameters()):
        if p.grad is not None:
            assert n not in grads, (n, grads)
            grads[n] = torch.norm(p.grad.detach(), norm_type)
            norms[n] = p.norm()
            shapes[n] = p.shape

    names = sorted(grads, key=lambda x: grads[x])
    for i, n in enumerate(names):
        if n in grads_table:
            delta_grad = (grads[n].item() - grads_table[n])
            delta_norm = (norms[n].item() - norms_table[n])
        else:
            delta_grad = delta_norm = 0
        grads_table[n] = grads[n].item()
        norms_table[n] = norms[n].item()
        if verbose:
            print(
                'param #{:03d}: {:35s}: \t shape={:20s}, \t grad norm={:.3f} (d={:.3f}), '
                '\t param norm={:.3f} (d={:.3f})'.format(
                    i,
                    '%35s' % n,
                    str(tuple(shapes[n])),
                    grads_table[n],
                    delta_grad,
                    norms[n],
                    delta_norm))

    grads = torch.stack(list(grads.values()))
    assert len(grads_table) == len(norms_table) == len(grads), (len(grads_table), len(norms_table), len(grads))
    total_grad_norm = torch.norm(grads, norm_type)
    total_norm = torch.norm(torch.stack(list(norms.values())), norm_type)
    print('{} params with gradients, total grad norm={:.3f}, total param norm={:.3f}\n'.format(
        tuple(grads.shape),
        total_grad_norm.item(),
        total_norm.item()))
    return


def transforms_imagenet(noise=False, cifar_style=False, im_size=224):
    """
    This is the same code as in https://github.com/facebookresearch/ppuda/blob/main/ppuda/vision/transforms.py#L88,
    but without ColorJitter to more closely reproduce ResNet-50 training results.
    :param noise:
    :param cifar_style:
    :param im_size:
    :return:
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = [
        transforms.RandomResizedCrop((32, 32) if cifar_style else im_size),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(  # remove this augmentation to more closely reproduce ResNet-50 training results
        #     brightness=0.4,
        #     contrast=0.4,
        #     saturation=0.4,
        #     hue=0.2),
        transforms.ToTensor(),
        normalize,
    ]
    if cifar_style:
        del train_transform[2]
    train_transform = transforms.Compose(train_transform)

    valid_transform = [
        transforms.Resize((32, 32) if cifar_style else max(im_size, 256)),
        transforms.CenterCrop(max(im_size, 224)),
        transforms.ToTensor()
    ]
    if cifar_style:
        del valid_transform[1]
    if noise:
        raise NotImplementedError('This transform is not expected during training. '
                                  'Use ppuda.vision.transforms.transforms_imagenet for evaluation on noisy images.')

    valid_transform.append(normalize)
    valid_transform = transforms.Compose(valid_transform)

    return train_transform, valid_transform
