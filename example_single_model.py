# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Predicts parameters of a PyTorch model using one of the pretrained GHNs.

Example (use --debug 1 to perform additional sanity checks and print more information):

    python example_single_model.py --ckpt ghn3tm8.pt --arch resnet50 --debug 0

"""


import torch
import torchvision
from ppuda.config import init_config

import ghn3
from ghn3 import from_pretrained, norm_check, Graph, Logger


# 1. Predict parameters of a PyTorch model using one of the pretrained GHNs
args = init_config(mode='eval')  # load arguments from the command line
assert args.arch is not None, ('architecture must be specified using, e.g. --arch resnet50', args.arch)

ghn = from_pretrained(args.ckpt, debug_level=args.debug).to(args.device)  # get a pretrained GHN

model = eval(f'torchvision.models.{args.arch}()').to(args.device)  # create a PyTorch model
model = ghn(model)  # predict parameters of the model

if args.debug:
    # optionally, check that the model is correctly predicted
    norm_check(model, arch=args.arch, ghn3_name=args.ckpt)

print('===Model %s can now be evaluated or fine-tuned on ImageNet or other dataset.===' % args.arch.upper())

if args.device == 'cuda':
    torch.cuda.empty_cache()

# 2. Example of fine-tuning the PyTorch model
n_steps = 10
print('\nExample of fine-tuning the %s model for %d steps:' % (args.arch.upper(), n_steps))
model.train()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
logger = Logger(n_steps)

for batch in range(n_steps):
    opt.zero_grad()
    loss = model(torch.randn(2, 3, 224, 224).to(args.device)).abs().mean()  # some dummy loss
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    opt.step()
    logger(batch, {'loss': loss.item(), 'grad norm': total_norm.item()})

if args.device == 'cuda':
    torch.cuda.empty_cache()

# 3. Example of fine-tuning the GHN on a single model
print('\nExample of fine-tuning the GHN for %d steps:' % n_steps)

model = eval(f'torchvision.models.{args.arch}()').to(args.device)
graph = Graph(model)  # create a graph of the model once so that it can be reused for all training iterations
model.train()
ghn.train()
ghn.debug_level = 0  # disable debug checks and prints for every forward pass to ghn
opt = torch.optim.SGD(ghn.parameters(), lr=0.1)
logger = Logger(n_steps)
for batch in range(n_steps):
    opt.zero_grad()
    model = ghn(model, graph, keep_grads=True)  # keep gradients of predicted params to backprop to the ghn
    loss = model(torch.randn(2, 3, 224, 224).to(args.device)).abs().mean()  # some dummy loss
    loss.backward()
    total_ghn_norm = torch.nn.utils.clip_grad_norm_(ghn.parameters(), 5)
    opt.step()
    logger(batch, {'loss': loss.item(), 'grad norm': total_ghn_norm.item()})
