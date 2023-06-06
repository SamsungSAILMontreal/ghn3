import os
import gc
import time
import psutil
import torch
import torchvision.transforms as transforms
import torch.distributed as dist
import ppuda.deepnets1m as deepnets1m
from ppuda.vision.imagenet import ImageNetDataset
from ppuda.vision.transforms import Noise
from torch.utils.data import DataLoader


process = psutil.Process(os.getpid())  # for reporting RAM usage


def named_layered_modules(model):

    if hasattr(model, 'module'):  # in case of multigpu model
        model = model.module
    layers = model._n_cells if hasattr(model, '_n_cells') else 1
    layered_modules = [{} for _ in range(layers)]
    cell_ind = 0
    for module_name, m in model.named_modules():

        cell_ind = m._cell_ind if hasattr(m, '_cell_ind') else cell_ind

        is_w = hasattr(m, 'weight') and m.weight is not None
        is_b = hasattr(m, 'bias') and m.bias is not None
        is_proj_w = hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None
        is_pos_enc = hasattr(m, 'pos_embedding') and m.pos_embedding is not None
        sz = None

        is_w = is_w or is_proj_w or is_pos_enc
        is_proj_b = hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None
        is_b = is_b or is_proj_b

        if is_w or is_b:
            if module_name.startswith('module.'):
                module_name = module_name[module_name.find('.') + 1:]

            if is_w:
                key = module_name + ('.in_proj_weight' if is_proj_w else ('.pos_embedding.weight'
                                                                          if is_pos_enc else '.weight'))
                w = m.in_proj_weight if is_proj_w else (m.pos_embedding if is_pos_enc else m.weight)
                layered_modules[cell_ind][key] = {'param_name': key, 'module': m, 'is_w': True,
                                                  'sz': sz if sz is not None else
                                                  (tuple(w) if isinstance(w, (list, tuple)) else w.shape)}
            if is_b:
                key = module_name + ('.in_proj_bias' if is_proj_b else '.bias')
                b = m.in_proj_bias if is_proj_b else m.bias
                layered_modules[cell_ind][key] = {'param_name': key, 'module': m, 'is_w': False,
                                                  'sz': tuple(b) if isinstance(b, (list, tuple)) else b.shape}

    return layered_modules


deepnets1m.net.named_layered_modules.__code__ = named_layered_modules.__code__


def transforms_imagenet(noise=False, cifar_style=False, im_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = [
        transforms.RandomResizedCrop((32, 32) if cifar_style else im_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
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
        valid_transform.append(Noise(0.08 if cifar_style else 0.18))
    valid_transform.append(normalize)
    valid_transform = transforms.Compose(valid_transform)

    return train_transform, valid_transform


def imagenet_loader(dataset='imagenet', data_dir='./data/', test=True, im_size=224,
                    batch_size=64, test_batch_size=64, num_workers=0, noise=False,
                    seed=1111, load_train_anyway=False):
    """

    For Inception-v3, the input size is 299x299, for all other models it is 224x224.
    """
    train_data = None
    train_transform, valid_transform = transforms_imagenet(noise=noise, cifar_style=False, im_size=im_size)
    imagenet_dir = os.path.join(data_dir, 'imagenet')

    if not test or load_train_anyway:
        train_data = ImageNetDataset(imagenet_dir, 'train', transform=train_transform, has_validation=not test)

    valid_data = ImageNetDataset(imagenet_dir, 'val', transform=valid_transform, has_validation=not test)

    shuffle_val = True  # to eval models with batch norm in the training mode (in case there is no running statistics)
    n_classes = len(valid_data.classes)
    generator = torch.Generator()
    generator.manual_seed(seed)  # to reproduce evaluation with shuffle=True on ImageNet

    print('loaded {}: {} classes, {} train samples (checksum={}), '
          '{} {} samples (checksum={:.3f})'.format(dataset,
                                                   n_classes,
                                                   train_data.num_examples if train_data else 'none',
                                                   ('%.3f' % train_data.checksum) if train_data else 'none',
                                                   valid_data.num_examples,
                                                   'test' if test else 'val',
                                                   valid_data.checksum))

    if train_data is None:
        train_loader = None
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)

    valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=shuffle_val,
                              pin_memory=True, num_workers=num_workers, generator=generator)

    return train_loader, valid_loader, n_classes


def is_ddp():
    return dist.is_available() and dist.is_initialized()


def get_ddp_rank():
    return dist.get_rank() if is_ddp() else 0


def clean_ddp():
    torch.cuda.empty_cache()
    gc.collect()
    print(f'rank {dist.get_rank()}, barrier enter', flush=True)
    dist.barrier()  # try to sync and throw a timeout error if not possible
    print(f'rank {dist.get_rank()}, destroy start', flush=True)
    dist.destroy_process_group()
    print('destroy finish', flush=True)
    torch.cuda.empty_cache()
    time.sleep(10)


def log(s, *args, **kwargs):
    if get_ddp_rank() == 0:
        print(s, *args, **kwargs)


class Logger:
    def __init__(self, max_steps):
        self.max_steps = max_steps
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
                   (time.time() - self.start_time) / max(1, (step + 1)),
                   process.memory_info().rss / 10 ** 9,
                   ('%.2f' % (torch.cuda.memory_reserved(0) / 10 ** 9)) if self.is_cuda else 'nan'),
            flush=True)
