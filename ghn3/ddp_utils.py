# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DDP utils.

"""


import os
import gc
import time
import torch
import torch.distributed as dist
from datetime import timedelta


def setup_ddp():
    """
    Setup distributed data parallel (ddp).
    :return: args with DDP attributes
    """

    class Args:
        pass
    args = Args()
    args.ddp = False
    # If torchrun is used, the environment variables RANK and WORLD_SIZE are set automatically so that ddp will be used.
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # DistributedDataParallel case
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        if not hasattr(args, 'device'):
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if args.device == 'cuda':
            args.device = f'cuda:{args.gpu}'
            torch.cuda.set_device(args.gpu)

        args.ddp = True
        dist.init_process_group(backend='nccl' if args.device.startswith('cuda') else 'gloo',
                                timeout=timedelta(minutes=30),  # increase timeout to avoid sync errors on some clusters
                                world_size=args.world_size,
                                rank=args.gpu)
        args.rank = dist.get_rank()
        if args.rank == 0:
            print(f"Start DDP with world size {args.world_size}, "
                  f"local rank {args.gpu}, "
                  f"master addr {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}.")
    else:
        args.rank = 0

    return args


def is_ddp():
    return dist.is_available() and dist.is_initialized()


def get_ddp_rank():
    return dist.get_rank() if is_ddp() else 0


def clean_ddp():
    """
    Used for debugging mainly. Attempting to clean the DDP so that training is resumed without errors.
    :return:
    """

    torch.cuda.empty_cache()
    gc.collect()
    print(f'rank {dist.get_rank()}, barrier enter', flush=True)
    dist.barrier()  # try to sync and throw a timeout error if not possible
    print(f'rank {dist.get_rank()}, destroy start', flush=True)
    dist.destroy_process_group()
    print('destroy finish', flush=True)
    torch.cuda.empty_cache()
    time.sleep(10)


def avg_ddp_metric(metric):
    """
    Computes average metric value gathered from workers.
    :param metric: input tensor
    :return: averaged tensor
    """
    lst = [torch.zeros_like(metric) for _ in range(dist.get_world_size())]
    dist.all_gather(lst, metric)
    avg = torch.stack(lst).mean().view_as(metric)
    return avg
