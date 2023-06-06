"""
Evaluates a GHN on all PyTorch models on ImageNet.

This script assumes the ImageNet dataset is already downloaded and stored at $SOME_DIR_TO_IMAGENET.
Assuming a GPU cluster node with $SLURM_TMPDIR as the data folder, a typical ImageNet data preparation is the following:
    ```
    cd $SLURM_TMPDIR;
    mkdir imagenet
    cd imagenet
    mkdir train
    echo "setting up imagenet data (can take a few minutes)..."
    tar -xf $SOME_DIR_TO_IMAGENET/ILSVRC2012_img_train.tar -C train/
    cd train
    for i in *.tar; do dir=${i%.tar}; mkdir -p $dir; tar xf $i -C $dir; done
    cp -r $SOME_DIR_TO_IMAGENET/val "$SLURM_TMPDIR/imagenet/"  # copy all validation images
    cp $SOME_DIR_TO_IMAGENET/ILSVRC2012_devkit_t12.tar.gz "$SLURM_TMPDIR/imagenet/";
    ```

Example:

    python eval_ghn_imagenet.py -d imagenet -D $SLURM_TMPDIR --ckpt ghn3xlm16.pt

"""


import torch
import torchvision
import torchvision.models as models
import time
import inspect
from ppuda.config import init_config
from ppuda.utils import infer, AvgrageMeter
from ghn3 import from_pretrained, get_metadata, imagenet_loader


args = init_config(mode='eval')

ghn = from_pretrained(args.ckpt, debug_level=args.debug).to(args.device)  # get a pretrained GHN
ghn.eval()  # should be a little bit more efficient in the eval mode

print('loading the %s dataset...' % args.dataset)
images_val, num_classes = imagenet_loader(args.dataset,
                                          args.data_dir,
                                          test=True,
                                          test_batch_size=64,
                                          num_workers=args.num_workers,
                                          noise=args.noise,
                                          im_size=224,
                                          seed=args.seed)[1:]

# Create a separate loader for 299x299 images required for inception_v3
images_val_im299 = imagenet_loader(args.dataset,
                                   args.data_dir,
                                   test=True,
                                   test_batch_size=64,
                                   num_workers=args.num_workers,
                                   noise=args.noise,
                                   im_size=299,
                                   seed=args.seed)[1]

assert ghn.num_classes == num_classes, 'the eval image dataset and the dataset that the GHN was trained must match'

norms = get_metadata(args.ckpt, attr='paramnorm')  # load meta-data for sanity checks

# Enumerate all PyTorch models of ImageNet classification
# Should be >= 74 models in torchvision>=0.13.1
all_torch_models = []
for m in dir(models):
    if m[0].isupper() or m.startswith('_') or m.startswith('get') or m == 'list_models' or \
            not inspect.isfunction(eval('models.%s' % m)):
        continue

    if m not in norms:
        print('=== %s was not in PyTorch at the moment of GHN-3 evaluation, so skipping it in this notebook ==='
              % m.upper())
        continue  # skip for consistency with the paper

    all_torch_models.append(m)

print('\n%d PyTorch models found. Predicting parameters for all...' % len(all_torch_models))

start_all = time.time()
norms_matched = []
top1_all = AvgrageMeter('std')  # use standard deviation (std) as the dispersion measure
for m_ind, m in enumerate(all_torch_models):

    try:
        # Predict parameters
        kw_args = {'init_weights': False} if m in ['googlenet', 'inception_v3'] else {}
        model = eval(f'models.{m}(**kw_args)').to(args.device)
        if not isinstance(model, torch.nn.Module):
            print('skipping %s, because it is not torch.nn.Module' % m)
            continue

        n_params = sum([p.numel() for p in model.parameters()]) / 10 ** 6
        if m == 'inception_v3':
            model.expected_input_sz = 299
            val_loader = images_val_im299
        else:
            val_loader = images_val

        print('\n{}/{}: {} with {:.2f}M parameters'.format(m_ind + 1,
                                                           len(all_torch_models),
                                                           m.upper(),
                                                           n_params), end='...')
        start = time.time()

        with torch.no_grad():  # to improve efficiency
            model = ghn(model)

        total_norm = torch.norm(torch.stack([p.norm() for p in model.parameters()]), 2)
        norms_matched.append(abs(norms[m] - total_norm.item()) < 1e-2)
        print('done in {:.2f} sec, total param norm={:.2f} ({})'.
              format(time.time() - start,
                     total_norm.item(),
                     'norms matched' if norms_matched[-1] else ('ERROR: norm not matched with %.2f' % norms[m])))
        # The `ERROR: norm not matched` can be fine if the model has some parameters not predicted by the GHN
        # and initialized randomly instead.

        print('Running evaluation for %s...' % m)
        val_loader.sampler.generator.manual_seed(args.seed)  # set the generator seed to reproduce results
        top1, top5 = infer(model.to(args.device), val_loader, verbose=True)
        top1_all.update(top1, 1)
    except Exception as e:
        print('ERROR for model %s: %s' % (m, e))
        continue

    # "WARNING: number of predicted ..." means that some layers in the model are not supported by the GHN
    # unsupported modules are initialized using built-in PyTorch methods

print(u'\nresults: (avg\u00B1std) top1={:.3f}\u00B1{:.3f}'.format(top1_all.cnt, top1_all.avg, top1_all.dispersion))
