# Copyright (c) 2023. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluates a GHN on one or all PyTorch models on ImageNet.
This script assumes the ImageNet dataset is already downloaded and set up as described in scripts/imagenet_setup.sh.

Example

    # Evaluating on all PyTorch models:
    python eval_ghn.py -d imagenet -D $SLURM_TMPDIR --ckpt ghn3xlm16.pt --split torch

    # Evaluating a single model like ResNet-50:
    python eval_ghn.py -d imagenet -D $SLURM_TMPDIR --ckpt ghn3xlm16.pt --arch resnet50 --split torch

    # Evaluating on all DeepNets1 models in the predefined split:
    python eval_ghn.py --ckpt ./checkpoints/ghn3tm8-c10-e833cce-1111/checkpoint.pt --split predefined
"""


import torch
import torchvision.models as models
import time
import argparse
import inspect
from ppuda.config import init_config
from ppuda.utils import infer, AvgrageMeter, adjust_net
from ppuda.vision.loader import image_loader
from ghn3 import from_pretrained, get_metadata, DeepNets1MDDP
from ghn3.ops import Network


parser = argparse.ArgumentParser(description='Evaluation of GHNs')
parser.add_argument('--save_ckpt', type=str, default=None,
                    help='checkpoint path to save the model with predicted parameters')
args = init_config(mode='eval', parser=parser, debug=0, split='torch')

ghn = from_pretrained(args.ckpt, debug_level=args.debug).to(args.device)  # get a pretrained GHN
ghn.eval()  # should be a little bit more efficient in the eval mode
is_imagenet = args.dataset.startswith('imagenet')
print('loading the %s dataset...' % args.dataset)
images_val, num_classes = image_loader(args.dataset,
                                       args.data_dir,
                                       test=True,
                                       test_batch_size=args.test_batch_size,
                                       num_workers=args.num_workers,
                                       noise=args.noise,
                                       im_size=args.imsize,
                                       seed=args.seed)[1:]

if args.arch in [None, 'inception_v3']:
    # Create a separate loader for 299x299 images required for inception_v3
    images_val_im299 = image_loader(args.dataset,
                                    args.data_dir,
                                    test=True,
                                    test_batch_size=args.test_batch_size,
                                    num_workers=args.num_workers,
                                    noise=args.noise,
                                    im_size=299,
                                    seed=args.seed)[1]

assert ghn.num_classes == num_classes, 'The eval image dataset and the dataset the GHN was trained on must match, ' \
                                       'But it is possible to fine-tune predicted parameters for a different dataset.' \
                                       'See example scripts for details.'

norms = get_metadata(args.ckpt, attr='paramnorm')  # load meta-data for sanity checks

is_torch = args.split == 'torch'
if is_torch:
    # Enumerate all PyTorch models of ImageNet classification
    # Should be >= 74 models in torchvision>=0.13.1
    models_queue = []
    for m in dir(models):
        if m[0].isupper() or m.startswith('_') or m.startswith('get') or m == 'list_models' or \
                not inspect.isfunction(eval('models.%s' % m)):
            continue

        if args.arch is not None and m == args.arch:
            models_queue = [m]
            break

        if norms is not None and m not in norms:
            print('=== %s was not in PyTorch at the moment of GHN-3 evaluation, so skipping it in this notebook ==='
                  % m.upper())
            continue  # skip for consistency with the paper

        models_queue.append(m)
    print('\n%d PyTorch models found. Predicting parameters for all...' % len(models_queue))

else:
    models_queue = DeepNets1MDDP.loader(meta_batch_size=1,
                                        dense=ghn.is_dense(),
                                        split=args.split,
                                        nets_dir=args.data_dir,
                                        arch=args.arch,
                                        virtual_edges=50 if ghn.ve else 1,
                                        large_images=is_imagenet,
                                        verbose=True,
                                        debug=args.debug > 0)

start_all = time.time()
norms_matched = []
top1_all = AvgrageMeter('std')  # use standard deviation (std) as the dispersion measure
for m_ind, m in enumerate(models_queue):

    try:
        # Predict parameters
        graphs = None
        if is_torch or (not is_torch and isinstance(m.net_args[0]['genotype'], str)):
            if not is_torch:
                graphs = m
                m = m.net_args[0]['genotype']
            kw_args = {'init_weights': False} if m in ['googlenet', 'inception_v3'] else {}
            model = eval(f'models.{m}(num_classes=num_classes, **kw_args)').to(args.device)
            if not isinstance(model, torch.nn.Module):
                print('skipping %s, because it is not torch.nn.Module' % m)
                continue
        else:
            model = Network(is_imagenet_input=is_imagenet,
                            num_classes=num_classes,
                            **m[0].net_args)
            graphs = m
            m = str(m[0].net_idx)

        if m == 'inception_v3':
            model.expected_input_sz = 299
            val_loader = images_val_im299
        else:
            model.expected_input_sz = args.imsize
            val_loader = images_val

        n_params = sum([p.numel() for p in model.parameters()]) / 10 ** 6
        print('\n{}/{}: {} with {:.2f}M parameters'.format(m_ind + 1,
                                                           len(models_queue),
                                                           m.upper(),
                                                           n_params), end='...')
        if args.device != 'cpu':
            torch.cuda.synchronize()
        start = time.time()

        if is_torch and not is_imagenet:
            model = adjust_net(model, large_input=False)  # adjust the model for small images such as 32x32 in CIFAR-10

        with torch.no_grad():  # to improve efficiency
            model = ghn(model, graphs=graphs, bn_track_running_stats=True, reduce_graph=True)  # predict parameters
            if args.save_ckpt is not None:
                torch.save({'state_dict': model.state_dict()}, args.save_ckpt)
                print('\nsaved the model with predicted parameters to {}\n'.format(args.save_ckpt))

            model.eval()  # set to the eval mode to disable dropout, etc.

            # set BN layers to the training mode to enable eval w/o running statistics
            def bn_set_train(module):
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.training = True
            model.apply(bn_set_train)

        total_norm = torch.norm(torch.stack([p.norm() for p in model.parameters()]), 2)
        if norms is not None:
            norms_matched.append(abs(norms[m] - total_norm.item()) < 1e-2)
        print('done in {:.2f} sec, total param norm={:.2f}{}'.
              format(time.time() - start,
                     total_norm.item(),
                     '' if norms is None else ' ({})'.format('norms matched' if norms_matched[-1] else
                                                             ('ERROR: norm not matched with %.2f' % norms[m]))))
        # The `ERROR: norm not matched` can be fine if the model has some parameters not predicted by the GHN
        # and initialized randomly instead.

        print('Running evaluation for %s...' % m)
        if is_imagenet:
            val_loader.sampler.generator.manual_seed(args.seed)  # set the generator seed to reproduce ImageNet results

        start = time.time()
        top1, top5 = infer(model.to(args.device), val_loader, verbose=False)
        print('\ntesting: top1={:.3f}, top5={:.3f} ({} eval samples, time={:.2f} seconds)'.format(
            top1, top5, val_loader.dataset.num_examples, time.time() - start), flush=True)
        top1_all.update(top1, 1)
    except Exception as e:
        print('ERROR for model %s: %s' % (m, e))

    # "WARNING: number of predicted ..." means that some layers in the model are not supported by the GHN
    # unsupported modules are initialized using built-in PyTorch methods

print(u'\nresults: (avg\u00B1std) top1={:.3f}\u00B1{:.3f}'.format(top1_all.cnt, top1_all.avg, top1_all.dispersion))
