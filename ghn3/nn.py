# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GHN-3 code.

"""


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math
import joblib
import time
import hashlib
from ppuda.deepnets1m.ops import PosEnc, NormLayers
from ppuda.ghn.nn import GHN, ShapeEncoder, ConvDecoder
from ppuda.utils import capacity
from .graph import Graph, GraphBatch
from huggingface_hub import hf_hub_download
from .utils import named_layered_modules, log


def from_pretrained(ghn3_name='ghn3xlm16.pt', **kwargs):
    """
    Loads a pretrained GHN-3 or GHN-2 model.
    :param ghn3_name: model name from https://huggingface.co/SamsungSAILMontreal/ghn3
    :param kwargs: GHN arguments
    :return: GHN model (in the training mode by default)
    """

    if ghn3_name is None or not ghn3_name.startswith('ghn'):
        raise ValueError(
            'GHN ckpt must be specified. Specify one from https://huggingface.co/SamsungSAILMontreal/ghn3.')

    verbose = 'debug_level' in kwargs and kwargs['debug_level']
    if verbose:
        log('loading %s...' % ghn3_name)

    is_ghn2 = False
    if ghn3_name.startswith('ghn3t'):
        hid = 64
        heads = 8
        layers = 3
    elif ghn3_name.startswith('ghn3s'):
        hid = 128
        heads = 16
        layers = 5
    elif ghn3_name.startswith('ghn3l'):
        hid = 256
        heads = 16
        layers = 12
    elif ghn3_name.startswith('ghn3xl'):
        hid = 384
        heads = 16
        layers = 24
    elif ghn3_name.startswith('ghn2'):
        hid = 32
        heads = 0
        layers = 1
        is_ghn2 = True
    else:
        raise NotImplementedError(ghn3_name)

    ghn = GHN3(hid=hid,
               max_shape=(hid * 2, hid * 2, 16, 16) if is_ghn2 else (hid, hid, 16, 16),
               num_classes=1000,
               heads=heads,
               layers=layers,
               is_ghn2=is_ghn2,
               weight_norm=True,
               ve=True,
               layernorm=True,
               **kwargs)

    state_dict = joblib.load(hf_hub_download(repo_id='SamsungSAILMontreal/ghn3', filename=ghn3_name))
    for n, p in state_dict.items():
        if n.find('decoder.') >= 0 and p.dim() == 4:
            state_dict[n] = p.squeeze()
    ghn.load_state_dict(state_dict)
    if verbose:
        log('loading %s with %d parameters is done!' % (ghn3_name,
                                                        sum([p.numel() for p in ghn.parameters()])))

    if not is_ghn2:
        # move the node embeddings for compatibility with GHN-2 code
        ghn.gnn[0].centrality_embed_in = ghn.centrality_embed_in
        ghn.gnn[0].centrality_embed_out = ghn.centrality_embed_out
        ghn.gnn[0].input_dist_embed = ghn.input_dist_embed
        del ghn.centrality_embed_in
        del ghn.centrality_embed_out
        del ghn.input_dist_embed

    return ghn


class GHN3(GHN):
    r"""
    Improved Transformer-based Graph HyperNetwork (GHN-3) based on the paper
    "Can We Scale Transformers to Predict Parameters of Diverse ImageNet Models?"
    (https://arxiv.org/abs/2303.04143)

    The key improvement is the usage of GraphormerLayers.

    Inherited from the GHN class (https://github.com/facebookresearch/ppuda/blob/main/ppuda/ghn/nn.py).
    But most of the functions are overridden to support GHN-3 and improve code.

    """
    def __init__(self, max_shape, num_classes, hid, heads, layers, is_ghn2=False, **kwargs):
        super().__init__(max_shape, num_classes, hid=hid, **kwargs)
        self._is_ghn2 = is_ghn2
        if not self._is_ghn2:
            self.gnn = SequentialMultipleInOut(*[
                GraphormerLayer(dim=hid,
                                num_heads=heads,
                                edge_dim=2 if layer == 0 else 0,
                                return_edges=layer < layers - 1) for layer in range(layers)])

            self.centrality_embed_in = nn.Embedding(self.gnn[0].max_degree + 1, hid)
            self.centrality_embed_out = nn.Embedding(self.gnn[0].max_degree + 1, hid)
            self.input_dist_embed = nn.Embedding(self.gnn[0].max_input_dist + 1, hid)

        self.decoder = ConvDecoder3(in_features=hid,
                                    hid=(hid * 4, hid * 8),
                                    out_shape=max_shape,
                                    num_classes=num_classes,
                                    is_ghn2=is_ghn2)
        self.shape_enc = ShapeEncoder3(hid=hid,
                                       num_classes=num_classes,
                                       max_shape=max_shape,
                                       debug_level=0)

        if not self._is_ghn2:
            # adjust initialization for GNN-3 models to improve training stability (especially for larger models)
            self.decoder_1d.fc[-2].apply(self._init_small)
            self.decoder.conv[-2].apply(self._init_small)
            self.decoder.class_layer_predictor[-1].apply(self._init_small)
            self.apply(self._init_embed)

    def forward(self,
                nets_torch,
                graphs=None,
                return_embeddings=False,
                predict_class_layers=True,
                bn_track_running_stats=True,
                keep_grads=False,
                reduce_graph=False):
        r"""
        Predict parameters for a list of >=1 networks.
        :param nets_torch: one network or a list of networks, each is based on nn.Module.
        :param graphs: Graph/GraphBatch object or None. If None, it will be constructed on the fly given the nets_torch.
        :param return_embeddings: True to return the node embeddings obtained after the last graph propagation step.
                                  return_embeddings=True is used for property prediction experiments.
        :param predict_class_layers: default=True predicts all parameters including the classification layers.
                                     predict_class_layers=False is used in fine-tuning experiments on a new task.
        :param bn_track_running_stats: bn_track_running_stats=True does not alter BN layer (used in fine-tuning).
                                       bn_track_running_stats=False sets track_running_stats of BN layers in nets_torch
                                       to False and sets the BN layers into the training mode,
                                       which is required for evaluation of predicted parameters w/o running stats.
        :param keep_grads: default=False to not keep the grads of the predicted params (used in evaluation/fine-tuning).
        :param reduce_graph: default=False to not remove redundant nodes in the graph (used in evaluation/fine-tuning).
        :return: nets_torch with predicted parameters and node embeddings if return_embeddings=True
        """

        device = self.embed.weight.device
        if graphs is None:
            graphs = GraphBatch([Graph(nets_torch, ve_cutoff=50 if self.ve else 1)],
                                dense=self.is_dense()).to_device(device)
        elif isinstance(graphs, Graph):
            graphs = GraphBatch([graphs], dense=self.is_dense()).to_device(device)

        if not graphs.on_device(device):
            graphs.to_device(device)

        assert graphs.dense == self.is_dense(), ('For this GHN architecture, '
                                                 'GraphBatch must be created with dense={}'.format(self.is_dense()))

        debug_info = self._init_debug_info(nets_torch, graphs)

        # Find mapping between embeddings and network parameters
        param_groups, params_map = self._map_net_params(graphs,
                                                        nets_torch,
                                                        reduce_graph=reduce_graph,
                                                        sanity_check=self.debug_level > 0)

        # Obtain initial embeddings for all nodes
        x = graphs.to_sparse(graphs.node_feat)[:, 0] if graphs.dense else graphs.node_feat[:, 0]
        x = self.shape_enc(self.embed(x), params_map, predict_class_layers=predict_class_layers)

        if self.is_dense():
            x = graphs.to_dense(x=x)[0]  # BNxC ->BxNxC
            mask = graphs.mask & graphs.mask.permute(0, 2, 1)  # B,N,N
        else:
            mask = graphs.node_feat[:, 1]

        # Update node embeddings using a GatedGNN, MLP or another model
        x = self.gnn(x, graphs.edges, mask)

        if self.layernorm:
            x = self.ln(x)

        # Predict max-sized parameters for a batch of nets using decoders
        if debug_info is not None:
            debug_info['n_tensors_pred'] = 0
            debug_info['n_params_pred'] = 0

        for key, inds in param_groups.items():
            if len(inds) == 0:
                continue
            x_ = x[torch.tensor(inds, device=x.device)]

            sz = key
            is_cls = False
            if len(sz) in [2, 3]:
                if len(sz) == 2 and sz[1] > 0:
                    # classification layer
                    w = self.decoder(x_, (sz[0], sz[1], 1, 1), class_pred=True)
                    is_cls = True
                else:
                    # 1d or cls-b
                    if len(sz) == 3:
                        w = self.decoder_1d(x_).view(len(inds), -1, 1, 1)
                    else:
                        w = self.decoder_1d(x_).view(len(inds), 2, -1)
                        if len(sz) == 2 and sz[1] < 0:
                            w = self.bias_class(w)
                            is_cls = True
            else:
                assert len(sz) == 4, sz
                w = self.decoder(x_, sz, class_pred=False)

            if not predict_class_layers and is_cls:
                continue  # do not set the classification parameters when fine-tuning

            # Transfer predicted parameters (w) to the networks
            for ind in inds:
                matched, _, w_ind = params_map[ind]

                if w_ind is None:
                    continue  # e.g. pooling

                m, sz, is_w = matched['module'], matched['sz'], matched['is_w']
                for it in range(2 if (len(sz) == 1 and is_w) else 1):

                    if len(sz) == 1:
                        # separately set for BN/LN biases as they are
                        # not represented as separate nodes in graphs
                        w_ = w[w_ind][1 - is_w + it]
                        if it == 1:
                            assert (type(m) in NormLayers and len(key) == 2 and key[1] == 0), \
                                (type(m), key)
                    else:
                        w_ = w[w_ind]

                    sz_set = self._set_params(m, self._tile_params(w_, sz), is_w=is_w & ~it, keep_grads=keep_grads)
                    if debug_info is not None:
                        debug_info['n_tensors_pred'] += 1
                        debug_info['n_params_pred'] += torch.prod(torch.tensor(sz_set))

        if bn_track_running_stats is None:
            bn_track_running_stats = self.training

        if not bn_track_running_stats:
            if self.debug_level:
                log('setting BN layers to the training mode to enable eval w/o running statistics')

            def bn_set_train(module):
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.training = True

            nets_torch.apply(bn_set_train)  # set BN layers to the training mode to enable eval w/o running statistics

        self._print_debug_info(nets_torch, debug_info)

        return (nets_torch, x) if return_embeddings else nets_torch

    def is_dense(self):
        return not self._is_ghn2

    def _init_debug_info(self, nets_torch, graphs):

        if self.debug_level:

            device = str(self.embed.weight.device)

            if not isinstance(nets_torch, (tuple, list)):
                nets_torch = [nets_torch]

            n_params = sum([capacity(net, is_grad=False)[1] for net in nets_torch])
            valid_ops = None
            if self.debug_level > 1:
                # This check performs backprop, so "Trying to backward through the graph a second time" can occur
                valid_ops = sum([graph.num_valid_nodes(net) for graph, net in zip(graphs, nets_torch)])
                log('\nnumber of learnable parameter tensors: {}, total number of parameters: {}'.format(
                    valid_ops, n_params))

            if device != 'cpu':
                torch.cuda.synchronize()
            return {'n_params': n_params, 'valid_ops': valid_ops, 'start_time': time.time(), 'device': device}
        return None

    def _print_debug_info(self, nets_torch, debug_info):

        if self.debug_level and debug_info is not None:
            device = debug_info['device']
            if device != 'cpu':
                torch.cuda.synchronize()  # to correctly measure the time on cuda

            log('number of parameter tensors predicted using GHN: {}, '
                'total parameters predicted: {} ({}), time to predict (on {}): {:.4f} sec'.format(
                 debug_info['n_tensors_pred'],
                 debug_info['n_params_pred'],
                 ('matched!' if debug_info['n_params'] == debug_info['n_params_pred'] else
                  'error! not matched with {} actual params'.format(debug_info['n_params'])).upper(),
                 device.upper(),
                 time.time() - debug_info['start_time']))

            if self.debug_level > 1:
                if debug_info['valid_ops'] != debug_info['n_tensors_pred']:
                    log('WARNING: number of learnable tensors ({}) must be the same as the '
                        'number of predicted tensors ({})'.format(debug_info['valid_ops'],
                                                                  debug_info['n_tensors_pred']))

            if self.debug_level > 2:
                if not isinstance(nets_torch, (tuple, list)):
                    nets_torch = [nets_torch]

                for net_id, net in enumerate(nets_torch):
                    log('\npredicted parameter stats for net %d:' % net_id)
                    for n, p in net.named_parameters():
                        log('{:30s} ({:30s}): min={:.3f} \t max={:.3f} \t mean={:.3f} \t std={:.3f} '
                            '\t norm={:.3f}'.format(
                             n[:30],
                             str(p.shape)[:30],
                             p.min().item(),
                             p.max().item(),
                             p.mean().item(),
                             p.std().item(),
                             torch.norm(p).item()))

    def _tile_params(self, w, target_shape):

        t, s = target_shape, w.shape

        if len(t) == 1:
            if len(s) == 1:
                w = w[:min(t[0], s[0])]
            elif len(s) == 2:
                w = w[:min(t[0], s[0]), 0]
            elif len(s) > 2:
                w = w[:min(t[0], s[0]), 0, w.shape[-2] // 2, w.shape[-1] // 2]
        elif len(t) == 2:
            if len(s) == 2:
                w = w[:min(t[0], s[0]), :min(t[1], s[1])]
            elif len(s) > 2:
                w = w[:min(t[0], s[0]), :min(t[1], s[1]), w.shape[-2] // 2, w.shape[-1] // 2]
        elif len(t) == 3:
            if len(s) == 3:
                w = w[:min(t[0], s[0]), :min(t[1], s[1]), :min(t[2], s[2])]
            elif len(s) > 3:
                # handle the positional encoding in vision transformers
                w = w.reshape(*s[:2], -1).permute(0, 2, 1)  # e.g. [1, 49, 384]
                w = w[:min(t[0], w.shape[0]), :min(t[1], w.shape[1]), :min(t[2], w.shape[2])]
                # add class token embedding initialized randomly since GHNs were trained on models without class token
                w = torch.cat((torch.normal(mean=0, std=0.02, size=(1, 1, w.shape[2]), device=w.device), w), dim=1)
        else:
            s2 = min(t[2], s[2]) if len(s) > 2 else 1
            s3 = min(t[3], s[3]) if len(s) > 3 else 1
            if len(s) > 2:
                if self._is_ghn2:
                    w = w[:min(t[0], s[0]), :min(t[1], s[1]), :s2, :s3]
                else:
                    offset = (w.shape[-2] // 2, w.shape[-1] // 2)
                    w = w[:min(t[0], s[0]), :min(t[1], s[1]),
                          offset[0] - s2 // 2: offset[0] + int(np.ceil(s2 / 2)),
                          offset[1] - s3 // 2: offset[1] + int(np.ceil(s3 / 2))]
            else:
                w = w[:min(t[0], s[0]), :min(t[1], s[1])].unsqueeze(2).unsqueeze(3)

        s = w.shape

        assert len(s) == len(t), (s, t)

        # Tile out_channels
        if t[0] > s[0]:
            n_out = int(np.ceil(t[0] / s[0]))
            if len(t) == 1:
                # w = w.expand(n_out, -1).reshape(n_out * w.shape[0])[:t[0]]
                w = w.repeat(n_out)[:t[0]]
            elif len(t) == 2:
                # w = w.expand(n_out, -1, -1).reshape(n_out * w.shape[0], w.shape[1])[:t[0]]
                w = w.repeat((n_out, 1))[:t[0]]
            else:
                # w = w.expand(n_out, -1, -1, -1, -1).reshape(n_out * w.shape[0], *w.shape[1:])[:t[0]]
                w = w.repeat((n_out, 1, 1, 1))[:t[0]]

        # Tile in_channels
        if len(t) > 1:
            if t[1] > s[1]:
                n_in = int(np.ceil(t[1] / s[1]))
                if len(t) == 2:
                    w = w.repeat((1, n_in))[:, :t[1]]
                else:
                    w = w.repeat((1, n_in, 1, 1))[:, :t[1]]
            elif len(t) == 3 and len(s) == 3 and t[2] > s[2]:
                n_in = int(np.ceil(t[2] / s[2]))
                w = w.repeat((1, 1, n_in))[:, :, :t[2]]

        # Chop out any extra bits tiled
        if len(t) == 1:
            w = w[:t[0]]
        elif len(t) == 2:
            w = w[:t[0], :t[1]]
        elif len(t) == 3:
            w = w[:t[0], :t[1], :t[2]]
        else:
            if self._is_ghn2:
                w = w[:t[0], :t[1], :t[2], :t[3]]
            else:
                offset = (w.shape[-2] // 2, w.shape[-1] // 2)
                w = w[:t[0], :t[1],
                      offset[0] - t[2] // 2: offset[0] + int(np.ceil(t[2] / 2)),
                      offset[1] - t[3] // 2: offset[1] + int(np.ceil(t[3] / 2))]

        return w

    def _set_params(self, module, tensor, is_w, keep_grads=None):
        r"""
        Copies the predicted parameter tensor to the appropriate field of the module object.
        :param module: nn.Module
        :param tensor: predicted tensor
        :param is_w: True if it is a weight, False if it is a bias
        :param keep_grads: True if the gradients of the predicted tensor should be kept (e.g. for training the GHN)
        :return: the shape of the copied tensor
        """
        if self.weight_norm:
            tensor = self._normalize(module, tensor, is_w)
        if isinstance(module, nn.MultiheadAttention):
            key = 'in_proj_weight' if is_w else 'in_proj_bias'
        elif isinstance(module, torchvision.models.vision_transformer.Encoder):
            key = 'pos_embedding'
        else:
            key = 'weight' if is_w else 'bias'
        target_param = getattr(module, key)
        sz_target = tuple(target_param) if isinstance(target_param, (list, tuple)) else target_param.shape
        if len(sz_target) == 4 and tensor.dim() == 2:
            tensor = tensor.unsqueeze(2).unsqueeze(3)  # for squeezenet

        keep_grads = self.training if keep_grads is None else keep_grads
        if keep_grads:
            if isinstance(target_param, (list, tuple)):
                if key == 'weight':
                    module.weight = tensor
                elif key == 'bias':
                    module.bias = tensor
                else:
                    raise NotImplementedError(key)

            else:
                module.__dict__[key] = tensor  # set the value avoiding the internal logic of PyTorch
                # update parameters, so that named_parameters() will return tensors
                # with gradients (for multigpu and other cases)
                module._parameters[key] = tensor
        else:
            assert isinstance(target_param, nn.Parameter), type(target_param)
            # copy to make sure there is no sharing of memory
            target_param.data = tensor.clone()

        set_param = getattr(module, key)
        assert sz_target == set_param.shape, (sz_target, set_param.shape, tensor.shape, key)
        return set_param.shape

    def _normalize(self, module, p, is_w):
        r"""
        Normalizes the predicted parameter tensor according to the Fan-In scheme described in the paper.
        :param module: nn.Module
        :param p: predicted tensor
        :param is_w: True if it is a weight, False if it is a bias
        :return: normalized predicted tensor
        """
        if p.dim() > 1:

            sz = p.shape

            if len(sz) > 2 and sz[2] >= 11 and sz[0] == 1:
                assert isinstance(module, (PosEnc, torchvision.models.vision_transformer.Encoder)), (sz, module,
                                                                                                     type(module))
                return p  # do not normalize positional encoding weights

            no_relu = len(sz) > 2 and (sz[1] == 1 or sz[2] < sz[3])
            if no_relu:
                # layers not followed by relu
                beta = 1.
            else:
                # for layers followed by rely increase the weight scale
                beta = 2.

            # fan-out:
            # p = p * (beta / (sz[0] * p[0, 0].numel())) ** 0.5

            # fan-in:
            p = p * (beta / p[0].numel()) ** 0.5

        else:

            if is_w:
                p = 2 * torch.sigmoid(0.5 * p)  # BN/LN norm weight is [0,2]
            else:
                p = torch.tanh(0.2 * p)  # bias is [-1,1]

        return p

    def _map_net_params(self, graphs, nets_torch, reduce_graph=None, sanity_check=False):
        r"""
        Matches the parameters in the models with the nodes in the graph.
        Performs additional steps.
        :param graphs: GraphBatch object
        :param nets_torch: a single neural network of a list
        :param reduce_graph: True to remove redundant computational nodes from the graph for faster training
        :param sanity_check: True for some extra checks
        :return: mapping, params_map
        """
        mapping = {}
        params_map = {}

        reduce_graph = self.training if reduce_graph is None else reduce_graph

        nets_torch = [nets_torch] if type(nets_torch) not in [tuple, list] else nets_torch
        for b, (node_info, net) in enumerate(zip(graphs.node_info, nets_torch)):

            target_modules = net.__dict__['_layered_modules'] if hasattr(net, '_layered_modules') \
                else named_layered_modules(net)

            param_ind = torch.sum(graphs.n_nodes[:b]).item()

            for cell_id in range(len(node_info)):
                for (node_ind, p_, name, sz, last_weight, last_bias) in node_info[cell_id]:

                    param_name = p_ if p_.endswith(
                        ('.weight', '.bias', 'in_proj_weight', 'in_proj_bias')) else p_ + '.weight'
                    try:
                        matched = [target_modules[cell_id][param_name]]
                    except:
                        matched = []

                    if len(matched) == 0:
                        if sz is not None:
                            params_map[param_ind + node_ind] = ({'sz': sz}, None, None)

                        if sanity_check:
                            for pattern in ['input', 'sum', 'concat', 'pool', 'glob_avg', 'msa', 'cse']:
                                good = name.find(pattern) >= 0
                                if good:
                                    break
                            if not good:
                                raise ValueError('\n'.join((
                                    '\nInvalid model/graph:',
                                    'cell_id: %d' % cell_id,
                                    'param_name: %s' % param_name,
                                    'name: %s' % name,
                                    '%d node_info: %s' % (len(node_info[cell_id]), node_info[cell_id]),
                                    '%d target_modules: %s' % (len(target_modules[cell_id]), target_modules[cell_id]),
                                    '\nThis error may be fixed by passing reduce_graph=False to the GHN forward pass, '
                                    'but it may result in slower training.')))
                    else:
                        sz = matched[0]['sz']

                        def min_sz(j):
                            # to group predicted shapes and improve parallelization and
                            # at the same time not to predict much more than needed
                            n = min(sz[j], self.max_shape[j])
                            if n % 3 == 0:
                                n = n // 3 * 4  # make multiple of 4 to be consistent with the decoder
                            if n >= self.max_shape[j] / 2:
                                n = self.max_shape[j]
                            return n

                        if len(sz) == 1:
                            key = (min_sz(0), -1) if last_bias else (min_sz(0), 0)
                        elif last_weight:
                            key = (min_sz(0), min_sz(1))
                        elif len(sz) == 2:
                            key = (min_sz(0), min_sz(1), 1, 1)
                        elif len(sz) == 3:
                            if sz[0] == 1 and min(sz[1:]) > 1:  # e.g. [1, 197, 768]
                                s = int(np.floor(sz[1] ** 0.5))  # [1, 197, 768] -> [1, 768, 14, 14]
                                key = (1, sz[2], s, s)
                            else:
                                key = (min_sz(0), min_sz(1), min_sz(2))  # e.g. layer_scale in ConvNeXt
                        else:
                            key = (min_sz(0), min_sz(1), sz[2], sz[3])

                        if key not in mapping:
                            mapping[key] = []
                        params_map[param_ind + node_ind] = (matched[0], key, len(mapping[key]))
                        mapping[key].append(param_ind + node_ind)
                        if reduce_graph:
                            del target_modules[cell_id][param_name]

                if reduce_graph:
                    # Prune redundant ops in Network by setting their params to None to speed up training
                    for m in target_modules[cell_id].values():
                        if m['is_w']:
                            m['module'].weight = None
                            if hasattr(m['module'], 'bias') and m['module'].bias is not None:
                                m['module'].bias = None

        return mapping, params_map

    def _init_small(self, module):
        """
        Reduces weight values to help avoid nan loss in some cases
        :param module:
        :return:
        """
        module.weight.data /= 5.0
        module.bias.data *= 0
        return

    def _init_embed(self, module):
        """
        Initializes embedding layers as commonly done in transformers
        :param module:
        :return:
        """
        if isinstance(module, nn.Embedding):
            d = module.weight.shape[1]
            nn.init.trunc_normal_(module.weight.data, std=d ** (-0.5))
        return


class ShapeEncoder3(ShapeEncoder):
    r"""
    Updated ShapeEncoder of GHN-2 to support more different PyTorch models.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, params_map, predict_class_layers=True):
        shape_ind = self.dummy_ind.repeat(len(x), 1)

        self.printed_warning = False
        for node_ind in params_map:
            sz = params_map[node_ind][0]['sz']
            if sz is None:
                continue

            sz_org = sz
            if len(sz) == 1:
                sz = (sz[0], 1)
            if len(sz) == 2:
                sz = (sz[0], sz[1], 1, 1)
            if len(sz) == 3:
                # Special treatment of 3D weights for some models like ViT.
                if sz[0] == 1 and min(sz[1:]) > 1:  # e.g. [1, 197, 768]
                    s = int(np.floor(sz[1] ** 0.5))
                    sz = (1, sz[2], s, s)
                else:
                    sz = (sz[0], sz[1], sz[2], 1)

            assert len(sz) == 4, sz

            if not predict_class_layers and params_map[node_ind][1] in ['cls_w', 'cls_b']:
                # keep the classification shape as though the GHN is used on the dataset it was trained on
                sz = (self.num_classes, *sz[1:])

            recognized_sz = 0
            for i in range(4):
                # if not in the dictionary, then use the maximum shape
                if i < 2:  # for out/in channel dimensions
                    shape_ind[node_ind, i] = self.channels_lookup[
                        sz[i] if sz[i] in self.channels_lookup else self.channels[-1]]
                    if self.debug_level and not self.printed_warning:
                        recognized_sz += int(sz[i] in self.channels_lookup_training)
                else:  # for kernel height/width
                    shape_ind[node_ind, i] = self.spatial_lookup[
                        sz[i] if sz[i] in self.spatial_lookup else self.spatial[-1]]
                    if self.debug_level and not self.printed_warning:
                        recognized_sz += int(sz[i] in self.spatial_lookup_training)

            if self.debug_level and not self.printed_warning:  # print a warning once per architecture
                if recognized_sz != 4:
                    print('WARNING: unrecognized shape %s, so the closest shape at index %s will be used instead.' % (
                        sz_org, ([self.channels[c.item()] if i < 2 else self.spatial[c.item()] for i, c in
                                  enumerate(shape_ind[node_ind])])))
                    self.printed_warning = True

        shape_embed = torch.cat(
            (self.embed_channel(shape_ind[:, 0]),
             self.embed_channel(shape_ind[:, 1]),
             self.embed_spatial(shape_ind[:, 2]),
             self.embed_spatial(shape_ind[:, 3])), dim=1)

        return x + shape_embed


class ConvDecoder3(ConvDecoder):
    """
    Updated ConvDecoder of GHN-2 with small changes for GHN-3 (specifically, using spatial offsets for generating higher
     quality weights).
     Also, nn.Linear is used instead of nn.Conv2d to help to resolve some mysterious RuntimeError CUDA errors on A100.

    """
    def __init__(self, is_ghn2=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_ghn2 = is_ghn2
        for dec_layer in [0, 2]:
            conv = self.conv[dec_layer]
            self.conv[dec_layer] = torch.nn.Linear(conv.weight.shape[1], conv.weight.shape[0])
        dec_layer = 1
        conv = self.class_layer_predictor[dec_layer]
        self.class_layer_predictor[dec_layer] = torch.nn.Linear(conv.weight.shape[1],
                                                                conv.weight.shape[0])

    def forward(self, x, max_shape=(1, 1, 1, 1), class_pred=False):

        N = x.shape[0]
        x = self.fc(x).view(N, -1, *self.out_shape[2:])
        if self._is_ghn2:
            x = x[:, :, :max_shape[2], :max_shape[3]]
        else:
            offset = self.out_shape[2] // 2
            x = x[:, :,
                  max(0, offset - max_shape[2] // 2): offset + int(np.ceil(max_shape[2] / 2)),
                  max(0, offset - max_shape[3] // 2): offset + int(np.ceil(max_shape[3] / 2))]

        out_shape = (*self.out_shape[:2], min(self.out_shape[2], max_shape[2]), min(self.out_shape[3], max_shape[3]))
        x = self.conv(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x.reshape(N, out_shape[0], out_shape[1], *out_shape[2:])
        x = x[:, :, :max_shape[1], :max_shape[2], :max_shape[3]]
        if min(max_shape[2:]) > min(out_shape[2:]):
            assert x.shape[0] == 1, x.shape
            x = F.interpolate(x[0], max_shape[2:], mode='bilinear').unsqueeze(0)

        if class_pred:
            assert x.shape[-2] == x.shape[-1], ('require squared weights at this point', x.shape)
            k = x.shape[-1] // 2
            x = self.class_layer_predictor(x[:, :, :, k, k].permute(0, 2, 1)).permute(0, 2, 1)  # N, num_classes, in
        else:
            x = x[:, :max_shape[0]]

        return x


""" 
Graphormer-based layers.

"""


class FeedForward(nn.Module):
    """
    Standard MLP applied after each self-attention in Transformer layers.

    """
    def __init__(self, in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop) if drop > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.net(x)


class EdgeEmbedding(nn.Module):
    """
    Simple embedding layer that learns a separate embedding for each edge value (e.g. 0 and 1 for binary edges).

    """
    def __init__(self, hid, max_len=5000):
        super().__init__()
        # Based on positional encoding in original Transformers (Attention is All You Need)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hid, 2) * (-math.log(10000.0) / hid))
        pe = torch.zeros(max_len, hid)  # first dim is batch
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.embed = nn.Embedding(max_len, hid)
        self.embed.weight.data = pe

    def forward(self, x):
        return self.embed(x)


class MultiHeadSelfAttentionEdges(nn.Module):
    """
    Multi-head self-attention layer with edge embeddings.

    When edge_dim=0, this is a standard multi-head self-attention layer.
    However, the edge features produced by the first MultiHeadSelfAttentionEdges layer are propagated
    to the subsequent layers.

    """
    def __init__(self, dim, edge_dim=0, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.dim = dim
        self.edge_dim = edge_dim
        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity())

        if self.edge_dim > 0:
            # assume 255+2 is the maximum shortest path distance in graphs
            self.edge_embed = EdgeEmbedding(dim, max_len=257)
            self.proj_e = nn.Sequential(nn.Linear(edge_dim * dim, dim),
                                        nn.ReLU(),
                                        nn.Linear(dim, num_heads))

            # the embedding layers below will be copied from the parent module (see the from_pretrained function)
            # self.centrality_embed_in = nn.Embedding(self.max_degree + 1, dim)
            # self.centrality_embed_out = nn.Embedding(self.max_degree + 1, dim)
            # self.input_dist_embed = nn.Embedding(self.max_input_dist + 1, dim)

    def forward(self, x, edges, mask=None):
        """
        MultiHeadSelfAttentionEdges forward pass.
        :param x: node features of the shape (B, N, C).
        :param edges: edges of shape (B, N, N, 2), where 2 is due to concatenating forward and backward edges.
        :param mask: mask of shape (B, N, N) with zeros for zero-padded edges and ones otherwise.
        :return: x of shape (B, N, C) and edges of shape (B, N, N, h),
        where h is the number of self-attention heads (8 by default).

        edges are propagated to the next layer, but are not going to be updated in the subsequent layers.

        """

        if self.edge_dim > 0:
            edges = self.edge_embed(edges)  # (B, N, N, 2) -> (B, N, N, 2, dim)
            edges = edges.reshape(*edges.shape[:-2], -1)  # (B, N, N, 2, dim) -> (B, N, N, 2*dim)
            edges = self.proj_e(edges)  # (B, N, N, 2*dim) -> (B, N, N, 8) -- 8 is the number of heads by default

        # standard multi-head self-attention
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # attention matrix attn is going to be of shape (B, 8, N, N),
        # so we permute the edges (B, N, N, 8) to (B, 8, N, N) before summing up them with attn
        # this operation results in the edge-aware attention matrix that is used to update node features
        attn = (q @ k.transpose(-2, -1)) * self.scale + edges.permute(0, 3, 1, 2)

        if mask is not None:
            # Zero-out attention values corresponding to zero-padded edges/nodes
            # based on https://discuss.pytorch.org/t/apply-mask-softmax/14212/25
            attn = attn.masked_fill(~mask.unsqueeze(1), -2 ** 15)  # 2**15 to work with amp

        # standard steps of self-attention layers
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.to_out(x)
        return x, edges


class SequentialMultipleInOut(nn.Sequential):
    """
    Wrapper to build a sequence of GraphormerLayers with multiple inputs.

    """

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input, *args):
        for module in self:
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input, *args)
        return input


class GraphormerLayer(nn.Module):
    """
    Graphormer layer taking node features x (B, N, C), directed graph edges (B, N, N) and mask (B, N, N) as inputs.
    B is a batch size corresponding to multiple architectures (for evaluation B=1).
    N is the maximum number of nodes in the batch of graphs.
    C is the dimension of the node features (dim).

    x are node features (e.g. produced by an embedding layer).
    We further augment x with centrality in/out and input_dist embeddings to enrich them with the graph structure.
    To correctly compute input_dist, edges must contain the shortest path distances b/w nodes as described below.

    edges is an adjacency matrix with values from 0 to 255. 0 means no edge, while values > 0 are edge distances.
    In a simple case, edges can be a binary matrix indicating which nodes are connected (1) and which are not (0).
    In GHN-3 we follow GHN-2 and use the shortest path distances (1, 2, 3, ...) between pairs of nodes as edge values.
    Note that edges are directed (correspond to the forward pass direction), so the edges matrix is upper triangular.

    mask is a binary mask indicating which nodes are valid (1) and which are zero-padded (0).
    """
    def __init__(self,
                 dim,
                 edge_dim=0,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=False,
                 act_layer=nn.GELU,
                 eps=1e-5,
                 return_edges=True):
        """

        :param dim: hidden size.
        :param edge_dim: GHN-3 only the first Graphormer layer has edge_dim>0 (we use edge_dim=2).
        For all other layers edge_dim=0, which corresponds to the vanilla Transformer layer.
        :param num_heads: number of attention heads.
        :param mlp_ratio: ratio of mlp hidden dim to embedding dim.
        :param qkv_bias: whether to add bias to qkv projection.
        :param act_layer: activation layer.
        :param eps: layer norm eps.
        :param return_edges: whether to return edges (for GHN-3 all but the last Graphormer layer returns edges).
        """
        super().__init__()

        self.return_edges = return_edges
        self.edge_dim = edge_dim
        self.max_degree = 100
        self.max_input_dist = 1000
        self.ln1 = nn.LayerNorm(dim, eps=eps)
        self.attn = MultiHeadSelfAttentionEdges(dim,
                                                edge_dim=edge_dim,
                                                num_heads=num_heads,
                                                qkv_bias=qkv_bias)
        self.ln2 = nn.LayerNorm(dim, eps=eps)
        self.ff = FeedForward(in_features=dim,
                              hidden_features=int(dim * mlp_ratio),
                              act_layer=act_layer)

    def forward(self, x, edges, mask=None):

        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        B, N, C = x.shape

        if self.edge_dim > 0:

            if edges.dim() == 2:
                # construct a dense adjacency matrix
                # if the edges are already of the shape (B, N, N), then do nothing
                edges_dense = torch.zeros(N, N).to(edges)
                edges_dense[edges[:, 0], edges[:, 1]] = edges[:, 2]
                edges = edges_dense.unsqueeze(0)

            # edges must be (B, N, N) at this stage
            edges_1hop = (edges == 1).long()
            x += self.centrality_embed_in(torch.clip(edges_1hop.sum(1), 0, self.max_degree))
            x += self.centrality_embed_out(torch.clip(edges_1hop.sum(2), 0, self.max_degree))
            x += self.input_dist_embed(torch.clip(edges[:, 0, :], 0, self.max_input_dist))

            if mask is not None:
                x = x * mask[:, :, :1]

            edges = torch.stack((edges, edges.permute(0, 2, 1)), dim=-1) + 2  # separate fw,bw edge embeddings

        x_attn, edges = self.attn(self.ln1(x), edges, mask)
        x = x + x_attn
        x = x + self.ff(self.ln2(x))

        return (x, edges, mask) if self.return_edges else x.reshape(B * N, C)


def norm_check(model, arch='resnet50', ghn3_name='ghn3xlm16.pt'):
    """
    Sanity check to make sure GHN works correctly.
    :param model: PyTorch model.
    :param arch: name of the architecture.
    :param ghn3_name: name of the GHN model.
    :return:
    """
    total_norm = torch.norm(torch.stack([p.norm() for p in model.parameters()]), 2).item()
    norm = get_metadata(ghn3_name, arch=arch, attr='paramnorm')
    log('Predicted params total norm={:.4f} ({})'.
        format(total_norm,
               ('check passed!' if abs(norm - total_norm) < 1e-2 else
                ('ERROR: norm check not matched with %.2f' % norm)) if norm else 'no norm check available'))
    # This error can be fine if the model has some parameters not predicted by the GHN and initialized randomly instead.


def get_metadata(ghn3_name='ghn3xlm16.pt', arch=None, attr=None):
    """
    Get metadata for the GHN models (for sanity checks) by reading the json file
    from https://huggingface.co/SamsungSAILMontreal/ghn3/blob/main/ghn3_results.json.
    Note that the same file is also located at https://github.com/SamsungSAILMontreal/ghn3/blob/main/ghn3_results.json.
    :param ghn3_name:
    :param arch:
    :param attr:
    :return:
    """
    if ghn3_name is not None:
        if ghn3_name == 'ghn3xlm16.pt':
            key = 'ghn3'
        elif ghn3_name == 'ghn3tm8.pt':
            key = 'ghn3-t'
        elif ghn3_name == 'ghn2.pt':
            key = 'ghn2'
        elif ghn3_name == 'randinit':
            key = 'randinit'
        else:
            log('WARNING: meta data not unavailable for %s' % ghn3_name)
            return None

    cache_file = hf_hub_download(repo_id='SamsungSAILMontreal/ghn3', filename='ghn3_results.json')
    with open(cache_file, 'rb') as f:
        # md5 check to make sure the file is not corrupted
        assert hashlib.md5(f.read()).hexdigest() == 'bb703997a56fa7f5359c8e6142524e72', 'corrupted ghn3_results.json'

    with open(cache_file, 'r') as f:
        # same file as https://github.com/SamsungSAILMontreal/ghn3/blob/main/ghn3_results.json
        meta_data = json.load(f)

    if ghn3_name is None:
        return meta_data

    meta_data_filtered = {}
    for a in meta_data:
        meta_data_filtered[a] = {}
        for k in meta_data[a]:
            if k.startswith('ghn3-t') and key == 'ghn3':
                continue
            if k.startswith(key):
                meta_data_filtered[a][k.split('-')[-1]] = float(meta_data[a][k])

    if arch is not None:
        meta_data_filtered = meta_data_filtered[arch]
        if attr is not None:
            meta_data_filtered = meta_data_filtered[attr]
    elif attr is not None:
        meta_data_filtered = {arch: meta_data_filtered[arch][attr] for arch in meta_data_filtered}

    return meta_data_filtered
