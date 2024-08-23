# Copyright (c) 2023. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains a PyTorch model on ImageNet. DistributedDataParallel (DDP) training is used if `torchrun` is used as shown below.
This script assumes the ImageNet dataset is already downloaded and set up as described in scripts/imagenet_setup.sh.

About using Graph HyperNetworks (GHNs) for initialization (ignore if training from scratch):
The model can be initialized with the GHN-3 by passing the `--ckpt ghn3xlm16.pt` flag.
Instead of ghn3xlm16.pt, other GHN-3 variants can be used. See README.md for details.
In case of initializing with GHN-3, the initial learning rate should be reduced.
For example, for ResNet-50, we use 0.025 instead of 0.1 (see the paper for details), which can be set by `--lr 0.025`.

Example:

    # To train ResNet-50 from scratch on ImageNet (single GPU, standard hyperparams):
    python train_ddp.py -d imagenet -D $SLURM_TMPDIR --arch resnet50 --name resnet50-randinit -e 90 --wd 1e-4 -b 128

    # To train ResNet-50 from the GHN-3 init on ImageNet (single GPU, hyperparams as in our paper):
    python train_ddp.py -d imagenet -D $SLURM_TMPDIR --arch resnet50 --name resnet50-ghn3init -e 90 --wd 1e-4 \
    -b 128 --lr 0.025 --ckpt ghn3xlm16.pt

    # Fancy setup*, from scratch: 4 GPUs (batch size=2024), DDP, automatic mixed precision, 50 epochs, LAMB optimizer:
    export OMP_NUM_THREADS=8
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train_ddp.py -d imagenet -D $SLURM_TMPDIR --arch resnet50 \
    --name resnet50-randinit-ddp-timm -e 50 --wd 1e-2 -b 512 --lr 8e-3 --amp --scheduler cosine-warmup --opt lamb \
    -i 160 --grad_clip 0 --label_smooth 0 --bce --timm_aug
    # results in 75.62±0.15 top-1 accuracy (average of 3 runs)

    # Fancy setup*, GHN-3 init, reduced lr: 4 GPUs (batch size=2024), DDP, automatic mixed precision, 50 ep, LAMB opt:
    export OMP_NUM_THREADS=8
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train_ddp.py -d imagenet -D $SLURM_TMPDIR --arch resnet50 \
    --name resnet50-ghn3init-ddp-timm -e 50 --wd 1e-3 -b 512 --lr 6e-3 --amp --scheduler cosine --opt lamb \
    -i 160 --grad_clip 0 --label_smooth 0 --bce --timm_aug --ckpt ghn3xlm16.pt
    # results in 75.83±0.11 top-1 accuracy (average of 3 runs)

    # *Based on the A3 training procedure from "ResNet strikes back: An improved training procedure in timm"
    # (https://arxiv.org/abs/2110.00476), but reduced to 50 epochs instead of 100.

    # Use eval.py to evaluate the trained model on ImageNet.

"""


import torchvision
import argparse
import time
from functools import partial
from ppuda.config import init_config
from ppuda.utils import capacity
from ppuda.vision.loader import image_loader
from ghn3 import log, Trainer, setup_ddp, transforms_imagenet, clean_ddp

log = partial(log, flush=True)


def main():
    parser = argparse.ArgumentParser(description='ImageNet training')
    parser.add_argument('-c', '--compile', type=str, default=None, help='use pytorch2.0 compilation for efficiency')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--bce', action='store_true', help='use BCE loss instead of cross-entropy')
    parser.add_argument('--timm_aug', action='store_true', help='use timm augmentations (RandAugment, Mixup, Cutmix)')
    parser.add_argument('--interm_epoch', type=int, default=5, help='intermediate epochs to keep checkpoints for')

    ddp = setup_ddp()
    args = init_config(mode='train_net', parser=parser, verbose=ddp.rank == 0, debug=0, beta=1e-5)
    # beta is the amount of noise added to params (if GHN is used for init, otherwise ignored), default: 1e-5

    log('loading the %s dataset...' % args.dataset.upper())
    train_queue = image_loader(args.dataset,
                               args.data_dir,
                               test=not args.val,
                               load_train_anyway=True,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               seed=args.seed,
                               ddp=ddp.ddp,
                               im_size=args.imsize,
                               transforms_train_val=transforms_imagenet(im_size=args.imsize, timm_aug=args.timm_aug),
                               verbose=ddp.rank == 0)[0]

    trainer = Trainer(eval(f'torchvision.models.{args.arch}()'),
                      opt=args.opt,
                      opt_args={'lr': args.lr, 'weight_decay': args.wd, 'momentum': args.momentum},
                      scheduler='mstep' if args.scheduler is None else args.scheduler,
                      scheduler_args={'milestones': args.lr_steps, 'gamma': args.gamma},
                      n_batches=len(train_queue),
                      grad_clip=args.grad_clip,
                      device=args.device,
                      log_interval=args.log_interval,
                      amp=args.amp,                       # automatic mixed precision (default: False)
                      label_smoothing=args.label_smooth,  # label smoothing (default: 0.1)
                      save_dir=args.save,
                      ckpt=args.ckpt,                     # GHN-3 init (default: None/from scratch)
                      epochs=args.epochs,
                      verbose=ddp.rank == 0,
                      bce=args.bce,
                      mixup=args.timm_aug,
                      compile_mode=args.compile,          # pytorch2.0 compilation for potential speedup (default: None)
                      beta=args.beta,
                      )

    log('\nStarting training {} with {} parameters!'.format(args.arch.upper(), capacity(trainer._model)[1]))

    for epoch in range(trainer.start_epoch, args.epochs):

        log('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, args.epochs, trainer.get_lr()))

        if ddp.ddp:
            # make sure sample order is different for each epoch and each seed
            if epoch == trainer.start_epoch:
                train_queue.sampler.seed = args.seed
            train_queue.sampler.set_epoch(epoch)
            log(f'shuffle {args.dataset} train loader: set seed to {args.seed}, epoch to {epoch}')

        trainer.reset_metrics(epoch)

        for step, (images, targets) in enumerate(train_queue, start=trainer.start_step):

            if step >= len(train_queue):  # if we resume training from some start_step > 0, then need to break the loop
                break

            trainer.update(images, targets)  # update model params
            trainer.log(step)

            if args.save:
                trainer.save(epoch, step, {'args': args}, interm_epoch=args.interm_epoch)  # save model checkpoint

        trainer.scheduler_step()  # lr scheduler step

    log('done at {}!'.format(time.strftime('%Y%m%d-%H%M%S')))
    if ddp.ddp:
        clean_ddp()


if __name__ == '__main__':
    main()
