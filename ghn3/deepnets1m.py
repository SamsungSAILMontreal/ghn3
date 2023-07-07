# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Loaders for DeepNets-1M supporting distributed training.

"""

import numpy as np
import torch.utils.data
import networkx as nx
import h5py
import os
from functools import partial
from torch.utils.data.distributed import DistributedSampler
from ppuda.utils import rand_choice
from ppuda.deepnets1m.genotypes import from_dict, PRIMITIVES_DEEPNETS1M
from ppuda.deepnets1m.loader import DeepNets1M, NetBatchSampler, MAX_NODES_BATCH
from .graph import Graph, GraphBatch
from .utils import log
from .ddp_utils import is_ddp
from .ops import NetworkLight


class DeepNets1MDDP(DeepNets1M):
    r"""
    DeepNets1M loader supporting DDP.
    """

    def __init__(self,
                 dense=True,
                 wider_nets=True,
                 debug=False,
                 **kwargs):
        if 'nets_dir' in kwargs and kwargs['nets_dir'] != './data':
            # Reset to a local ./data folder if hdf5 is not found in nets_dir (handles some complicated cluster setups)
            nets_dir = kwargs['nets_dir']
            split = kwargs['split'] if 'split' in kwargs else 'train'
            deepnets_file = 'deepnets1m_%s.hdf5' % (split if split in ['train', 'search'] else 'eval')
            h5_file = os.path.join(nets_dir, deepnets_file)
            if not os.path.exists(h5_file):
                kwargs['nets_dir'] = './data'
            log('DeepNets1MDDP nets_dir set to %s as deepnets1m files not found at %s' % (kwargs['nets_dir'], nets_dir))

        super(DeepNets1MDDP, self).__init__(**kwargs)
        self.wider_nets = wider_nets
        self.dense = dense
        self.debug = debug

        if self.split != 'predefined':

            # Improve efficiency of reading metadata so that __getitem__ is faster
            self.primitives_dict = {op[:4]: i for i, op in enumerate(PRIMITIVES_DEEPNETS1M)}
            assert len(self.primitives_dict) == len(PRIMITIVES_DEEPNETS1M), (len(self.primitives_dict),
                                                                             len(PRIMITIVES_DEEPNETS1M))

            def dict_to_list(d):
                d_lst = [None for _ in range(1 + max(list(map(int, list(d.keys())))))]
                for k, v in d.items():
                    d_lst[int(k)] = v
                return d_lst

            self.primitives_ext = dict_to_list(self.primitives_ext)
            self.op_names_net = dict_to_list(self.op_names_net)


    @staticmethod
    def loader(meta_batch_size=1, dense=True, **kwargs):
        nets = DeepNets1MDDP(dense=dense, **kwargs)
        sampler = NetBatchSamplerDDP(nets, meta_batch_size) if nets.is_train else None
        n_w = (0 if meta_batch_size <= 1 else min(8, max(4, meta_batch_size // 2))) if nets.is_train else 0
        log('num workers', n_w)
        loader = torch.utils.data.DataLoader(nets,
                                             batch_sampler=sampler,
                                             batch_size=1,
                                             pin_memory=False,
                                             collate_fn=partial(GraphBatch, dense=dense),
                                             num_workers=n_w)
        return (loader, sampler) if nets.is_train else loader  # need to return sampler for distributed training

    def __getitem__(self, idx):

        if self.split == 'predefined':
            graph = self.nets[idx]
        else:

            if self.h5_data is None:  # A separate fd is opened for each worker process
                self.h5_data = h5py.File(self.h5_file, mode='r')

            args = self.nets[idx]
            idx = self.h5_idx[idx] if self.h5_idx is not None else idx
            cell, n_cells = from_dict(args['genotype']), args['n_cells']

            args['imagenet_stride'] = 4

            if self.is_train:

                is_conv_dense = sum([n[0] in ['conv_5x5', 'conv_7x7'] for n in
                                     cell.normal + cell.reduce]) > 0

                # if wider_nets, the sampled networks are on average wider and can have lower imagenet_stride
                # this helps improve generalization to larger models at test time

                # if wider_nets, use cifar10's number of parameters for both cifar10 and imagenet
                # (results in slightly larger nets for imagenet)
                num_params = args['num_params']['imagenet' if self.large_images and
                                                not self.wider_nets else 'cifar10'] / 10 ** 6

                # attempt to use a smaller stride for imagenet to reduce the bias to stride = 4
                if self.wider_nets and self.large_images and args['glob_avg'] and args['stem_type'] == 0 and \
                        args['stem_pool'] and not (num_params > 0.2 or n_cells > 8 or is_conv_dense):
                    # glob_avg must be used to use a smaller stride otherwise the number of features can be too high
                    # if the network consumes a lot of memory, then allow for a small stride only if stem_pool is used
                    args['imagenet_stride'] = np.random.choice([2, 4])

                fc = rand_choice(self.fc_dim, 4)  # 64-256
                if num_params > (2.0 if self.wider_nets else 0.8) or not args['glob_avg'] or is_conv_dense or \
                        n_cells > (14 if self.wider_nets else 12):
                    C = self.num_ch.min()                                       # 32
                elif num_params > 0.4 or n_cells > 10:
                    C = rand_choice(self.num_ch, 4 if self.wider_nets else 2)   # [32, 48, 64, 80] or [32, 48]
                elif num_params > 0.2 or n_cells > 8:
                    C = rand_choice(self.num_ch, 5 if self.wider_nets else 3)   # [32, 48, 64, 80, 96] or [32, 48, 64]
                else:
                    C = rand_choice(self.num_ch)                                # [32, 48, 64, 80, 96, 112, 128]
                    if C <= 64:
                        fc = rand_choice(self.fc_dim)

                args['C'] = C.item()
                args['fc_dim'] = fc.item()

            net_args = {'genotype': cell}
            for key in ['norm', 'ks', 'preproc', 'glob_avg', 'stem_pool', 'C_mult',
                        'n_cells', 'fc_layers', 'C', 'fc_dim', 'stem_type',
                        'imagenet_stride']:
                if key == 'C' and self.split == 'wide':
                    net_args[key] = args[key] * (2 if self.large_images else 4)
                else:
                    net_args[key] = args[key]

            graph = self._init_graph(self.h5_data[self.split][str(idx)]['adj'][()],
                                     self.h5_data[self.split][str(idx)]['nodes'][()],
                                     net_args)
            graph.net_idx = idx
            if self.is_train and not self.debug:
                graph.net = NetworkLight(is_imagenet_input=self.large_images,
                                         num_classes=1000 if self.large_images else 10,
                                         **net_args)

        return graph

    def _init_graph(self, A, nodes, net_args):
        """
        This function fixes a few graph construction bugs in the original code.
        """

        layers = net_args['n_cells']
        is_vit = sum([n[0] == 'msa' for n in net_args['genotype'].normal + net_args['genotype'].reduce]) > 0
        N = A.shape[0]
        assert N == len(nodes), (N, len(nodes))

        recompute_ve = False  # recompute virtual edges if the graph is fixed

        if net_args['stem_type'] == 1 and not is_vit:

            if net_args['norm'] is not None:
                stem0, stem1 = 4, 6
                if self.debug:
                    assert self.op_names_net[nodes[stem0][2]] == 'stem0.4.weight', (
                        self.op_names_net[nodes[stem0][2]], net_args)
                    assert self.op_names_net[nodes[stem1][2]] == 'stem1.2.weight', (
                        self.op_names_net[nodes[stem1][2]], net_args)
            else:
                stem0, stem1 = 2, 3
                if self.debug:
                    assert self.op_names_net[nodes[stem0][2]] == 'stem0.3', (
                        self.op_names_net[nodes[stem0][2]], net_args)
                    assert self.op_names_net[nodes[stem1][2]] == 'stem1.1', (
                        self.op_names_net[nodes[stem1][2]], net_args)

            stem0_out = np.nonzero(A[stem0, :] == 1)[0]  # [stem1, cells.0.preproc0]
            stem1_out = np.nonzero(A[stem1, :] == 1)[0]  # [cells.0.preproc1, cell.1.preproc0]

            if len(stem1_out) == 1 and len(stem0_out) > 1:
                if stem0_out[-1] - stem0_out[-2] > 1:  # to avoid rewiring conv_1, conv_2
                    A[stem0, stem0_out[-1]] = 0  # disconnect s0 from cell 1
                    A[stem1, stem0_out[-1]] = 1  # connect s1 to cell 1
                    recompute_ve = True

        nodes_with_twoplus_in = np.nonzero((A == 1).sum(0) > 1)[0]
        for i in nodes_with_twoplus_in:
            name = self.primitives_ext[nodes[i][0]]
            if name not in ['concat', 'sum', 'cse']:
                incoming = np.nonzero(A[:, i] == 1)[0]
                A[incoming[1:], i] = 0
                recompute_ve = True

        if recompute_ve:
            A = self.recompute_virtual_edges(A)

        node_feat = torch.empty(N, 1, dtype=torch.long)
        node_info = [[] for _ in range(layers)]
        param_shapes = []
        for node_ind, node in enumerate(nodes):
            name = self.primitives_ext[node[0]]
            name_op_net = self.op_names_net[node[2]]

            cell_ind = node[1]

            sz = None

            if not name_op_net.startswith('classifier'):
                # fix some inconsistency between names in different versions of our code
                if (name_op_net.find('.to_qkv') or name_op_net.find('.to_out')) and name_op_net.find('attn.') < 0:
                    name_op_net = name_op_net.replace('to_qkv', 'attn.to_qkv').replace('to_out', 'attn.to_out')

                if len(name_op_net) == 0:
                    name_op_net = 'input'
                elif name_op_net.endswith('to_out.0.'):
                    name_op_net += 'weight'
                else:
                    parts = name_op_net.split('.')
                    for i, s in enumerate(parts):
                        if s == '_ops' and parts[i + 2] != 'op':
                            try:
                                _ = int(parts[i + 2])
                                parts.insert(i + 2, 'op')
                                name_op_net = '.'.join(parts)
                                break
                            except:
                                continue

                name_op_net = 'cells.%d.%s' % (cell_ind, name_op_net)

                stem_p = name_op_net.find('stem')
                pos_enc_p = name_op_net.find('pos_enc')
                if stem_p >= 0:
                    name_op_net = name_op_net[stem_p:]
                elif pos_enc_p >= 0:
                    name_op_net = name_op_net[pos_enc_p:]
                elif name.find('pool') >= 0:
                    sz = (1, 1, 3, 3)  # assume all pooling layers are 3x3 in our DeepNets-1M

            if name.startswith('conv_'):
                if name == 'conv_1x1':
                    sz = (16, 3, 1, 1)      # just some random shape for visualization purposes
                name = 'conv'               # remove kernel size info from the name
            elif name.find('conv_') > 0 or name.find('pool_') > 0:
                name = name[:len(name) - 4]  # remove kernel size info from the name
            elif name == 'fc-b':
                name = 'bias'

            param_shapes.append(sz)
            node_feat[node_ind] = self.primitives_dict[name[:4]]

            if name.find('conv') >= 0 or name.find('pool') >= 0 or name in ['bias', 'bn', 'ln', 'pos_enc']:
                node_info[cell_ind].append((node_ind, name_op_net, name, sz,
                                            node_ind == len(nodes) - 2, node_ind == len(nodes) - 1))

        A = torch.tensor(A, dtype=torch.long)
        A[A > self.virtual_edges] = 0

        graph = Graph(node_feat=node_feat, node_info=node_info, A=A, dense=self.dense, net_args=net_args)
        graph._param_shapes = param_shapes

        return graph

    def recompute_virtual_edges(self, A):
        if self.virtual_edges > 1:
            A[A > 1] = 0
            length = dict(nx.all_pairs_shortest_path_length(nx.DiGraph(A), cutoff=self.virtual_edges))
            for node1 in length:
                for node2 in length[node1]:
                    if length[node1][node2] > 0 and A[node1, node2] == 0:
                        A[node1, node2] = length[node1][node2]
        return A


class NetBatchSamplerDDP(NetBatchSampler):
    r"""
    NetBatchSampler that works with DistributedSampler.
    """

    def __init__(self, deepnets, meta_batch_size=1):
        super(NetBatchSampler, self).__init__(
            (DistributedSampler(deepnets) if is_ddp() else torch.utils.data.RandomSampler(deepnets))
            if deepnets.is_train
            else torch.utils.data.SequentialSampler(deepnets),
            meta_batch_size,
            drop_last=False)
        self.max_nodes_batch = int(
            MAX_NODES_BATCH / 8 * max(8, meta_batch_size)) if deepnets.is_train and meta_batch_size > 1 else None
        log('max_nodes_batch', self.max_nodes_batch, 'meta_batch_size', meta_batch_size)

    def check_batch(self, batch):
        return (self.max_nodes_batch is None or
                (self.sampler.dataset if is_ddp() else self.sampler.data_source).nodes[batch].sum() <=
                self.max_nodes_batch)

    def __iter__(self):
        epoch = 0
        while True:  # infinite sampler
            if is_ddp():
                log(f'shuffle DeepNets1MDDP train loader: set seed to {self.sampler.seed}, epoch to {epoch}')
                self.sampler.set_epoch(epoch)
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    if self.check_batch(batch):
                        yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                if self.check_batch(batch):
                    yield batch
            epoch += 1
