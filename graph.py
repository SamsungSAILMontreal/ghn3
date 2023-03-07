# Adjusted graph.py to support graph construction in all PyTorch models
# based on https://github.com/facebookresearch/ppuda/blob/main/ppuda/deepnets1m/graph.py


import numpy as np
import copy
import torch
import torch.nn as nn
import networkx as nx
import torchvision
from torchvision.models import SwinTransformer, SqueezeNet, VisionTransformer
from ppuda.deepnets1m.ops import NormLayers, PosEnc
from ppuda.deepnets1m.net import get_cell_ind
from ppuda.deepnets1m.genotypes import PRIMITIVES_DEEPNETS1M

import sys
sys.setrecursionlimit(10000)  # for large models like efficientnet_v2_l

t_long = torch.long


class Graph():
    r"""
    Container for a computational graph of a neural network.

    Example:

        graph = Graph(torchvision.models.resnet50())

    """

    def __init__(self, model=None, node_feat=None, node_info=None, A=None, edges=None, net_args=None, net_idx=None,
                 ve_cutoff=50, list_all_nodes=False):
        r"""
        :param model: Neural Network inherited from nn.Module
        """

        assert node_feat is None or model is None, 'either model or other arguments must be specified'

        self.model = model
        self._list_all_nodes = list_all_nodes  # True in case of dataset generation
        self.nx_graph = None  # NetworkX DiGraph instance

        if model is not None:
            sz = model.expected_input_sz if hasattr(model, 'expected_input_sz') else 224   # assume ImageNet image width/heigh by default
            self.expected_input_sz = sz if isinstance(sz, (tuple, list)) else (3, sz, sz)  # assume images by default
            self.n_cells = self.model._n_cells if hasattr(self.model, '_n_cells') else 1
            self._build_graph()   # automatically construct an initial computational graph
            self._add_virtual_edges(ve_cutoff=ve_cutoff)  # add virtual edges
            self._construct_features()  # initialize torch.Tensor node and edge features
            # self.visualize(figname='graph_swin.pdf', with_labels=True, font_size=4)
        else:
            self.n_nodes = len(node_feat)
            self.node_feat = node_feat
            self.node_info = node_info

            if edges is None:
                if not isinstance(A, torch.Tensor):
                    A = torch.from_numpy(A).long()
                ind = torch.nonzero(A)
                self.edges = torch.cat((ind, A[ind[:, 0], ind[:, 1]].view(-1, 1)), dim=1)
            else:
                self.edges = edges

        self.net_args = net_args
        self.net_idx = net_idx


    def num_valid_nodes(self, model=None):
        r"""
        Counts the total number of learnable parameter tensors.
        The function aims to find redundant parameter tensors that are disconnected from the computational graph.
        The function if based on computing gradients and, thus, is not reliable for all architectures.
        :param model: nn.Module based object
        :return: total number of learnable parameter tensors
        """
        if model is None:
            model = self.model
            expected_input_sz = self.expected_input_sz
        else:
            sz = model.expected_input_sz if hasattr(model, 'expected_input_sz') else 224
            expected_input_sz = sz if isinstance(sz, (tuple, list)) else (3, sz, sz)

        device = list(model.parameters())[0].device  # assume all parameters on the same device
        loss = model((torch.rand(1, *expected_input_sz, device=device) - 0.5) / 2)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss = loss.mean()
        if torch.isnan(loss):
            print('could not estimate the number of learnable parameter tensors due the %s loss', str(loss))
            return -1
        else:
            loss.backward()
            valid_params, valid_ops = 0, 0
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    assert p.grad is not None and p.dim() > 0, (name, p.grad)
                    s = p.grad.abs().sum()
                    if s > 1e-20:
                        valid_params += p.numel()
                        valid_ops += 1

        return valid_ops  # valid_params,


    def _build_graph(self):
        r"""
        Constructs a graph of a neural network in the automatic way.
        This function is written based on Sergey Zagoruyko's https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py (MIT License)
        PyTorch 1.9+ is required to run this script correctly for some architectures.
        Currently, the function is not written very clearly and may be improved.

        Issue: for Pytorch's torch.nn.Transformer the graph does not look correct at the moment.
        """

        param_map = {id(weight): (name, module) for name, (weight, module) in self._named_modules().items()}
        nodes, edges, seen = {}, [], {}

        def get_attr(fn):
            attrs = dict()
            for attr in dir(fn):
                if not attr.startswith('_saved_'):
                    continue
                val = getattr(fn, attr)
                attr = attr[len('_saved_'):]
                if torch.is_tensor(val):
                    attrs[attr] = "[saved tensor]"
                elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
                    attrs[attr] = "[saved tensors]"
                else:
                    attrs[attr] = str(val)
            return attrs

        def traverse_graph(fn):
            assert not torch.is_tensor(fn)
            if fn in seen:
                return seen[fn]

            fn_name = str(type(fn).__name__)
            node_link, link_start = None, None
            if fn_name.find('AccumulateGrad') < 0:
                leaf_nodes = []
                for u in fn.next_functions:
                    for i_u, uu in enumerate(u):
                        if uu is not None:  # so it's okay to keep uu=u[0] since u[1] never has variable field
                            if hasattr(uu, 'variable'):
                                var = uu.variable
                                try:
                                    name, module = param_map[id(var)]
                                except:
                                    print(uu, fn_name, var.size())
                                    raise

                                is_convnext = (name.find('.layer_scale') >= 0 and
                                               torchvision.models.convnext.CNBlock not in MODULES)
                                is_vit_torch = isinstance(module, VisionTransformer)
                                is_swin_torch = (name.find('relative_position_bias_table') >= 0)
                                if (type(module) in NormLayers and name.find('.bias') >= 0) or is_convnext or \
                                        is_vit_torch or is_swin_torch:
                                    continue  # do not add biases of NormLayers as nodes

                                leaf_nodes.append({'id': uu,
                                                   'param_name': name,
                                                   'attrs': {'size': var.size(), **get_attr(var)},
                                                   'module': module})

                                assert len(uu.next_functions) == 0

                if len(leaf_nodes) == 0:
                    leaf_nodes.append({'id': fn,
                                       'param_name': fn_name,
                                       'attrs': get_attr(fn),
                                       'module': None})

                assert not hasattr(fn, 'variable'), fn.variable

                for leaf in leaf_nodes:
                    node_link = str(id(leaf['id']))
                    if link_start is None:
                        link_start = node_link

                    seen[leaf['id']] = (node_link, leaf['param_name'])
                    nodes[node_link] = {
                                  'param_name': leaf['param_name'],
                                  'attrs': leaf['attrs'],
                                  'module': leaf['module']}

            seen[fn] = (node_link, fn_name)

            # recurse
            if hasattr(fn, 'next_functions'):
                for u in fn.next_functions:
                    for uu in u:
                        if uu is not None and not isinstance(uu, int):
                            link_, name_ = traverse_graph(uu)
                            if link_ is not None and link_start != link_:
                                edges.append((link_start, link_) if name_.find('bias') >= 0 else (link_, link_start))

            return node_link, fn_name

        var = self.model(torch.randn(2, *self.expected_input_sz))
        if not isinstance(var, (tuple, list, dict)):
            var = [var]
        if isinstance(var, dict):
            var = list(var.values())
        for v in var:
            if v is not None:
                traverse_graph(v.grad_fn)  # populate nodes and edges

        nodes_lookup = { key: i  for i, key in enumerate(nodes) }
        nodes = [ {'id': key, **nodes[key]} for key in nodes_lookup ]
        A = np.zeros((len(nodes), len(nodes)))  # +1 for the input node added below
        for out_node_id, in_node_id in edges:
            A[nodes_lookup[out_node_id], nodes_lookup[in_node_id]] = 1

        self._Adj = A
        self._nodes = nodes
        A, nodes = self._filter_graph()

        def fix_weight_bias():
            # Fix fc layers nodes and edge directions
            for pattern in ['weight']: #, 'bias']:  #  bias for pytorch's transformer layer
                for i, node in enumerate(nodes):
                    if (A[:, i].sum() == 0 and  # no incoming nodes
                            node['param_name'].find(pattern) >= 0): #and A[i, :].sum() <= 2):  # no more than 2 outcoming edges (not stable for all cases)
                        param_name_sfx = node['param_name'][:node['param_name'].rfind('.')]
                        for out_neigh in np.where(A[i, :])[0]:  # all nodes where there is an edge from i, e.g. bias or msa

                            if not (nodes[out_neigh]['param_name'].startswith(param_name_sfx)
                                    or nodes[out_neigh]['param_name'].lower().find('softmax') >= 0
                                    or (nodes[out_neigh]['param_name'].lower().find('add') >= 0 and A[i, :].sum() == 2)):  # to handle DETR and pytorch's multihead attention when the key != query
                                continue  # parameter name should be the same otherwise rewiring can go wrong

                            in_out = np.where(A[:, out_neigh])[0]   # incoming to the bias or msa
                            in_out_conn, in_out_bias = [], []
                            for j in in_out:
                                if i == j:
                                    continue
                                if nodes[j]['param_name'].startswith(param_name_sfx):
                                    in_out_bias.append(j)
                                    continue
                                if A[i, j] != 0: # or nodes[j]['param_name'].find('bias') < 0:
                                    continue  # if incoming to the bias does not have incoming edges and it's a weight or there is and edge from i (weight) to the bias, then don't do anything
                                in_out_conn.append(j)
                            in_out = np.array(in_out_conn)
                            in_out_bias = np.array(in_out_bias)
                            if len(in_out) == 0:
                                continue
                            A[in_out, i] = 1  # rewire edges coming to out_neigh (bias, AddB0) to node i (weight)
                            A[:, out_neigh] = 0  # remove all edges to out_neigh (bias or msa) from the nodes that have outcoming edges
                            if len(in_out_bias) > 0:
                                A[i, in_out_bias] = 1
                                A[in_out_bias, out_neigh] = 1  # keep the edge from i to out_neigh
                            else:
                                A[i, out_neigh] = 1  # keep the edge from i to out_neigh
                            A[i, i] = 0  # remove loop

        fix_weight_bias()

        for pattern in ['softmax']:
            for i, node in enumerate(nodes):
                if (node['param_name'].lower().find(pattern) >= 0):
                    for out_neigh in np.where(A[i, :])[0]:  # all nodes following node i (msa)
                        in_out = np.where(A[:, out_neigh])[0]  # nodes coming to out_neigh
                        # remove all edges coming to the node next to msa
                        # except the edge from msa to the next node
                        A[in_out, out_neigh] = 0  # rewire edges
                        A[i, out_neigh] = 1
                        A[i, i] = 0  # remove loop

        if self.model is not None and isinstance(self.model, SwinTransformer):
            for i, node in enumerate(nodes):
                if node['param_name'].lower().endswith('norm.weight'):
                    for out_neigh in np.where(A[i, :])[0]:
                        if nodes[out_neigh]['param_name'].endswith('norm1.weight'):
                            A[i, out_neigh] = 0
                elif node['param_name'].lower().endswith('attn.proj.bias'):
                    for out_neigh in np.where(A[i, :])[0]:
                        if nodes[out_neigh]['param_name'].endswith('reduction.weight'):
                            A[i, out_neigh] = 0  # from attn.bias to reduction.weight
                            for out_neigh2 in np.where(A[out_neigh, :])[0]:
                                # print(node['param_name'], nodes[out_neigh2]['param_name'])
                                if nodes[out_neigh2]['param_name'].startswith('AddBackward'):
                                    A[i, out_neigh2] = 1  # from attn.bias to reduction.weight

        A, nodes = self._filter_graph(unsupported_modules=set(['Add', 'Cat']))

        # Add input node
        try:
            A = np.pad(A, ((0, 1), (0, 1)), mode='constant')
            nodes.append({'id': 'input', 'param_name': 'input', 'attrs': None, 'module': None})
            # Should work for multiple inputs
            for ind in np.where(A.sum(0) == 0)[0]:  # nodes that do not have any incoming edges
                if nodes[ind]['param_name'].find('weight') >= 0:
                    A[-1, ind] = 1
        except Exception as e:
            print('!!! ERROR: adding input node failed', e)

        # Sort nodes in a topological order consistent with forward propagation
        try:
            A[np.diag_indices_from(A)] = 0
            ind = np.array(list(nx.topological_sort(nx.DiGraph(A))))
            nodes = [nodes[i] for i in ind]
            A = A[ind, :][:, ind]
        except Exception as e:
            print('!!! ERROR: topological sort failed', e)

        # Adjust graph for Transformers to be consistent with our original code
        for i, node in enumerate(nodes):
            if isinstance(node['module'], (PosEnc, torchvision.models.vision_transformer.Encoder)):
                nodes.insert(i + 1, { 'id': 'sum_pos_enc', 'param_name': 'AddBackward0', 'attrs': None, 'module': None })
                A = np.insert(A, i, 0, axis=0)
                A = np.insert(A, i, 0, axis=1)
                A[i, i + 1] = 1  # pos_enc to sum

        if self.model is not None and isinstance(self.model, SqueezeNet):
            assert nodes[-1]['param_name'].startswith('MeanBackward'), nodes[-1]
            assert nodes[-3]['param_name'].startswith('classifier'), nodes[-3]
            nodes.insert(len(nodes) - 3, copy.deepcopy(nodes[-1]))
            del nodes[-1]

        self._Adj = A
        self._nodes = nodes

        return


    def _filter_graph(self, unsupported_modules=None):
        r"""
        Remove redundant/unsupported nodes from the automatically constructed graphs.
        :return:
        """

        # These ops will not be added to the graph
        if unsupported_modules is None:
            unsupported_modules = set()
            for i, node in enumerate(self._nodes):
                ind = node['param_name'].find('Backward')
                name = node['param_name'][:len(node['param_name']) if ind == -1 else ind]
                # print(name)
                supported = False
                for key, value in MODULES.items():
                    if isinstance(key, str):
                        continue
                    if isinstance(node['module'], key):
                        supported = True
                        break
                if not supported and name not in MODULES:
                    if (name.find('AsStrided') < 0 and #not name.endswith('.layer_scale') and
                            'size' in node['attrs'] and len(node['attrs']['size']) > 0):
                        continue
                    unsupported_modules.add(node['param_name'])

            # Add ops requiring extra checks before removing
            unsupported_modules = ['Mul'] + list(unsupported_modules) + \
                                  (['Mean', 'Cat'] if self.model is not None and isinstance(self.model, SwinTransformer)
                                   else ['Mean', 'Add', 'Cat'])

        for pattern in unsupported_modules:

            ind_keep = []

            for i, node in enumerate(self._nodes):
                op_name, attrs = node['param_name'], node['attrs']

                if op_name.find(pattern) >= 0:

                    keep = False

                    if op_name.startswith('Mean'):
                        # Avoid adding mean operations (in CSE)
                        if not self._nodes[i - 1]['param_name'].startswith('classifier') and \
                                not self._nodes[i + 1]['param_name'].startswith('classifier') and \
                                not self._nodes[i - 1]['param_name'].startswith('fc') and \
                                not self._nodes[i + 1]['param_name'].startswith('fc') and \
                                not self._nodes[i - 1]['param_name'].startswith('head') and \
                                not self._nodes[i + 1]['param_name'].startswith('head'):
                            keep = False
                        elif isinstance(attrs, dict) and 'keepdim' in attrs:
                            keep = attrs['keepdim'] == 'True' or \
                                   self._nodes[i - 2]['param_name'].startswith('classifier') or \
                                   self._nodes[i - 1]['param_name'].startswith('classifier') or \
                                   self._nodes[i - 2]['param_name'].startswith('fc') or \
                                   self._nodes[i - 1]['param_name'].startswith('fc') or \
                                   self._nodes[i - 2]['param_name'].startswith('head') or \
                                   self._nodes[i - 1]['param_name'].startswith('head')
                        else:
                            # In pytorch <1.9 the computational graph may be inaccurate
                            keep = i < len(self._nodes) - 1 and not self._nodes[i + 1]['param_name'].startswith('cells.')

                    elif op_name.startswith('Mul'):
                        keep = self._nodes[i - 2]['param_name'].lower().startswith('hard') or \
                               self._nodes[i + 1]['param_name'].lower().find('sigmoid') >= 0    # CSE op

                    elif op_name.startswith('Cat') or op_name.startswith('Add'):        # Concat and Residual (Sum) ops
                        keep = len(np.where(self._Adj[:, i])[0]) > 1  # keep only if > 1 edges are incoming

                    if not keep:
                        # rewire edges from/to the to-be-removed node to its neighbors
                        for n1 in np.where(self._Adj[i, :])[0]:
                            for n2 in np.where(self._Adj[:, i])[0]:
                                if n1 != n2:
                                    self._Adj[n2, n1] = 1
                else:
                    keep = True

                if keep:
                    ind_keep.append(i)

            ind_keep = np.array(ind_keep)

            if len(ind_keep) < self._Adj.shape[0]:
                self._Adj = self._Adj[:, ind_keep][ind_keep, :]
                self._nodes = [self._nodes[i] for i in ind_keep]

        return self._Adj, self._nodes


    def _add_virtual_edges(self, ve_cutoff=50):
        r"""
        Add virtual edges with weights equal the shortest path length between the nodes.
        :param ve_cutoff: maximum shortest path length between the nodes
        :return:
        """

        self.n_nodes = len(self._nodes)

        assert self._Adj[np.diag_indices_from(self._Adj)].sum() == 0, (
            'no loops should be in the graph', self._Adj[np.diag_indices_from(self._Adj)].sum())

        # Check that the graph is connected and all nodes reach the final output
        self._nx_graph_from_adj()
        # length = nx.shortest_path(self.nx_graph, target=self.n_nodes - 1)
        # for node in range(self.n_nodes):
        #     if node not in length:
        #         print('WARNING: not all nodes reach the final node', node, self._nodes[node])

        # Check that all nodes have a path to the input
        # length = nx.shortest_path(self.nx_graph, source=0)
        # for node in range(self.n_nodes):
        #     if not (node in length or self._nodes[node]['param_name'].startswith('pos_enc') or
        #             self._nodes[node]['param_name'].find('position_bias') >= 0):
        #         print('WARNING: not all nodes have a path to the input', node)  # , self._nodes[node]

        if ve_cutoff > 1:
            length = dict(nx.all_pairs_shortest_path_length(self.nx_graph, cutoff=ve_cutoff))
            for node1 in length:
                for node2 in length[node1]:
                    if length[node1][node2] > 0 and self._Adj[node1, node2] == 0:
                        self._Adj[node1, node2] = length[node1][node2]
            assert (self._Adj > ve_cutoff).sum() == 0, ((self._Adj > ve_cutoff).sum(), ve_cutoff)
        return self._Adj


    def _construct_features(self):
        r"""
        Construct pytorch tensor features for nodes and edges.
        :return:
        """

        self.n_nodes = len(self._nodes)
        self.node_feat = torch.empty(self.n_nodes, 1, dtype=t_long)
        self.node_info = [[] for _ in range(self.n_cells)]
        self._param_shapes = []

        primitives_dict = {op: i for i, op in enumerate(PRIMITIVES_DEEPNETS1M)}

        n_glob_avg = 0
        cell_ind = 0
        for node_ind, node in enumerate(self._nodes):

            param_name = node['param_name']
            cell_ind_ = get_cell_ind(param_name, self.n_cells)
            if cell_ind_ is not None:
                cell_ind = cell_ind_

            pos_stem = param_name.find('stem')
            pos_pos = param_name.find('pos_enc')
            if pos_stem >= 0:
                param_name = param_name[pos_stem:]
            elif pos_pos >= 0:
                param_name = param_name[pos_pos:]

            if node['module'] is not None:

                # Preprocess param_name to be consistent with the DeepNets dataset
                parts = param_name.split('.')
                for i, s in enumerate(parts):
                    if s == '_ops' and parts[i + 2] != 'op':
                        try:
                            _ = int(parts[i + 2])
                            parts.insert(i + 2, 'op')
                            param_name = '.'.join(parts)
                            break
                        except:
                            continue

                name = MODULES[type(node['module'])](node['module'], param_name)

            else:
                ind = param_name.find('Backward')
                name = MODULES[param_name[:len(param_name) if ind == -1 else ind]]
                n_glob_avg += int(name == 'glob_avg')
                if self.n_cells > 1:
                    # Add cell id to the names of pooling layers, so that they will be matched with proper modules in Network
                    if param_name.startswith('MaxPool') or param_name.startswith('AvgPool'):
                        param_name = 'cells.{}.'.format(cell_ind) + name

            sz = None
            attrs = node['attrs']
            if isinstance(attrs, dict):
                if 'size' in attrs:
                    sz = attrs['size']
                elif name.find('pool') >= 0:
                    if 'kernel_size' in attrs:
                        sz = (1, 1, *[int(a.strip('(').strip(')').strip(' ')) for a in attrs['kernel_size'].split(',')])
                    else:
                        # Pytorch 1.9+ is required to correctly extract pooling attributes, otherwise the default pooling size of 3 is used
                        sz = (1, 1, 3, 3)
            elif node['module'] is not None:
                sz = (node['module'].weight if param_name.find('weight') >= 0 else node['module'].bias).shape

            if sz is not None:
                if len(sz) == 3 and sz[0] == 1 and min(sz[1:]) > 1:  # [1, 197, 768]
                    # print('WARNING: setting a 4d size instead of 3d', sz)
                    s = int(np.ceil(sz[1] ** 0.5))
                    sz = (1, sz[2], s, s)
                elif len(sz) == 4 and node_ind == len(self._nodes) - 2 and max(sz[2:]) == 1:
                    # print('WARNING: setting a 2d size instead of 4d', sz)
                    sz = sz[:2]

            self._param_shapes.append(sz)
            self.node_feat[node_ind] = primitives_dict[name]
            if node['module'] is not None or name.find('pool') >= 0 or self._list_all_nodes:
                self.node_info[cell_ind].append(
                    [node_ind,
                     param_name if node['module'] is not None else name,
                     name,
                     sz,
                     node_ind == len(self._nodes) - 2,
                     node_ind == len(self._nodes) - 1])

        if n_glob_avg != 1:
            print(
                'WARNING: n_glob_avg should be 1 in most architectures, but is %d in this architecture' %
                n_glob_avg)

        self._Adj = torch.tensor(self._Adj, dtype=t_long)

        ind = torch.nonzero(self._Adj)  # rows, cols
        self.edges = torch.cat((ind, self._Adj[ind[:, 0], ind[:, 1]].view(-1, 1)), dim=1)
        return


    def _named_modules(self):
        r"""
        Helper function to automatically build the graphs.
        :return:
        """
        modules = {}
        for n, m in self.model.named_modules():
            for np, p in m.named_parameters(recurse=False):
                if p is None:
                    continue
                key = n + '.' + np
                if key in modules:
                    assert id(p) == id(modules[key][0]), (n, np, p.shape, modules[key][0].shape)
                    continue
                modules[key] = (p, m)

        n_params = len(list(self.model.named_parameters()))
        assert len(modules) == n_params, (len(modules), n_params)

        return modules


    def _nx_graph_from_adj(self, A=None, remove_ve=True):
        """
        Creates NetworkX directed graph instance that is used for visualization, virtual edges and graph statistics.
        :return: nx.DiGraph
        """
        A = self._Adj if A is None else A
        A = A.data.cpu().numpy() if isinstance(A, torch.Tensor) else A
        if remove_ve:
            A[A > 1] = 0  # remove any virtual edges for the visualization/statistics
        else:
            A = A.astype(np.float32)
            ind = A > 1
            A[ind] = 1. / A[ind]
        self.nx_graph = nx.DiGraph(A)
        return self.nx_graph


    def properties(self, undirected=True, key=('avg_degree', 'avg_path')):
        """
        Computes graph properties.
        :param undirected: ignore edge direction when computing graph properties.
        :param key: a tuple/list of graph properties to estimate.
        :return: dictionary with property names and values.
        """
        G = self._nx_graph_from_adj()
        if undirected:
            G = G.to_undirected()
        props = {}
        for prop in key:
            if prop == 'avg_degree':
                degrees = dict(G.degree())
                assert len(degrees) == self._Adj.shape[0] == self.n_nodes, 'invalid graph'
                props[prop] = sum(degrees.values()) / self.n_nodes
            elif prop == 'avg_path':
                props[prop] = nx.average_shortest_path_length(G)
            else:
                raise NotImplementedError(prop)

        return props


    def visualize(self, node_size=50, figname=None, figsize=None, with_labels=False, vis_legend=False, label_offset=0.001, font_size=10,
                  remove_ve=True):
        r"""
        Shows the graphs/legend as in the paper using matplotlib.
        :param node_size: node size
        :param figname: file name to save the figure in the .pdf and .png formats
        :param figsize: (width, height) for a figure
        :param with_labels: show node labels (operations)
        :param vis_legend: True to only visualize the legend (graph will be ignored)
        :param label_offset: positioning of node labels when vis_legend=True
        :param font_size: font size for node labels, used only when with_labels=True
        :return:
        """

        import matplotlib.pyplot as plt
        from matplotlib import cm as cm

        self._nx_graph_from_adj(A=self._Adj, remove_ve=remove_ve)

        # first are conv layers, so that they have a similar color
        primitives_order = [2, 3, 4, 10, 5, 6, 11, 12, 13, 0, 1, 14, 7, 8, 9]
        assert len(PRIMITIVES_DEEPNETS1M) == len(primitives_order), 'make sure the lists correspond to each other'

        n_primitives = len(primitives_order)
        color = lambda i: cm.jet(int(np.round(255 * i / n_primitives)))
        primitive_colors = { PRIMITIVES_DEEPNETS1M[ind_org] : color(ind_new)  for ind_new, ind_org in enumerate(primitives_order) }
        # manually adjust some colors for better visualization
        primitive_colors['bias'] = '#%02x%02x%02x' % (255, 0, 255)
        primitive_colors['msa'] = '#%02x%02x%02x' % (10, 10, 10)
        primitive_colors['ln'] = '#%02x%02x%02x' % (255, 255, 0)

        node_groups = {'bn':        {'style': {'edgecolors': 'k',       'linewidths': 1,    'node_shape': 's'}},
                       'conv1':     {'style': {'edgecolors': 'k',       'linewidths': 1,    'node_shape': '^'}},
                       'bias':      {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 'd'}},
                       'pos_enc':   {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 's'}},
                       'ln':        {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 's'}},
                       'max_pool':  {'style': {'edgecolors': 'k',       'linewidths': 1,    'node_shape': 'o', 'node_size': 1.75 * node_size}},
                       'glob_avg':  {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 'o', 'node_size': 2 * node_size}},
                       'concat':    {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': '^'}},
                       'input':     {'style': {'edgecolors': 'k',       'linewidths': 1.5,  'node_shape': 's', 'node_size': 2 * node_size}},
                       'other':     {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 'o'}}}

        for group in node_groups:
            node_groups[group]['node_lst'] = []
            if 'node_size' not in node_groups[group]['style']:
                node_groups[group]['style']['node_size'] = node_size

        labels, node_colors = {}, []

        if vis_legend:
            node_feat = torch.cat((torch.tensor([n_primitives]).view(-1, 1),
                                   torch.tensor(primitives_order)[:, None]))
            param_shapes = [(3, 3, 1, 1)] + [None] * n_primitives
        else:
            node_feat = self.node_feat
            param_shapes = self._param_shapes

        for i, (x, sz) in enumerate(zip(node_feat.view(-1), param_shapes)):

            name = PRIMITIVES_DEEPNETS1M[x] if x < n_primitives else 'conv'

            labels[i] = self._nodes[i]['param_name'].replace('features', 'f').replace('.weight', '.w').replace('.bias', '.b')  # name[:20] if x < n_primitives else 'conv_1x1'
            node_colors.append(primitive_colors[name])

            if name.find('conv') >= 0 and sz is not None and \
                    ((len(sz) == 4 and np.prod(sz[2:]) == 1) or len(sz) == 2):
                node_groups['conv1']['node_lst'].append(i)
            elif name in node_groups:
                node_groups[name]['node_lst'].append(i)
            else:
                node_groups['other']['node_lst'].append(i)

        if vis_legend:
            fig = plt.figure(figsize=(20, 3) if figsize is None else figsize)
            G = nx.DiGraph(np.diag(np.ones(n_primitives), 1))
            pos = {j: (3 * j * node_size, 0) for j in labels }
            pos_labels = {j: (x, y - label_offset) for j, (x, y) in pos.items()}
        else:
            fig = plt.figure(figsize=(10, 10) if figsize is None else figsize)
            G = self.nx_graph
            pos = nx.drawing.nx_pydot.graphviz_layout(G)
            pos_labels = pos

        for node_group in node_groups.values():
            nx.draw_networkx_nodes(G, pos,
                                   node_color=[node_colors[j] for j in node_group['node_lst']],
                                   nodelist=node_group['node_lst'],
                                   **node_group['style'])
        if with_labels:
            nx.draw_networkx_labels(G, pos_labels, labels, font_size=font_size)

        nx.draw_networkx_edges(G, pos,
                               node_size=node_size,
                               width=0 if vis_legend else 1,
                               arrowsize=10,
                               alpha=0 if vis_legend else 1,
                               edge_color='white' if vis_legend else 'k',
                               arrowstyle='-|>')

        plt.grid(False)
        plt.axis('off')
        if figname is not None:
            plt.savefig(figname + '_%d.pdf' % 0, dpi=fig.dpi)
            plt.savefig(figname + '_%d.png' % 0, dpi=fig.dpi, transparent=True)
        else:
            plt.show()


def get_conv_name(module, param_name):
    if param_name.find('bias') >= 0:
        return 'bias'
    elif isinstance(module, nn.Conv2d) and module.groups > 1:
        return 'dil_conv' if min(module.dilation) > 1 else 'sep_conv'
    return 'conv'


# Supported modules
MODULES = {
            nn.Conv2d: get_conv_name,
            nn.Linear: get_conv_name,
            torch.nn.modules.linear.NonDynamicallyQuantizableLinear: get_conv_name,
            nn.BatchNorm2d: lambda module, param_name: 'bn',
            nn.LayerNorm: lambda module, param_name: 'ln',
            torchvision.models.convnext.LayerNorm2d: lambda module, param_name: 'ln',
            PosEnc: lambda module, param_name: 'pos_enc',
            torchvision.models.vision_transformer.Encoder: lambda module, param_name: 'pos_enc',
            torch.nn.modules.activation.MultiheadAttention: get_conv_name,
            torchvision.models.swin_transformer.ShiftedWindowAttention: lambda module, param_name: 'pos_enc',
            'input': 'input',
            'Mean': 'glob_avg',
            'AdaptiveAvgPool2D': 'glob_avg',
            'MaxPool2DWithIndices': 'max_pool',
            'AvgPool2D': 'avg_pool',
            'Softmax': 'msa',
            'Mul': 'cse',
            'Add': 'sum',
            'Cat': 'concat',
        }
