# Adjusted graph.py to support graph construction of all PyTorch models
# based on https://github.com/facebookresearch/ppuda/blob/main/ppuda/deepnets1m/graph.py


import numpy as np
import copy
import heapq
import torch
import torch.nn as nn
import networkx as nx
import transformers
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import ppuda.deepnets1m.ops as ops
from torch.nn.parallel.scatter_gather import Scatter as _scatter
from ppuda.deepnets1m.net import get_cell_ind
from ppuda.deepnets1m.genotypes import PRIMITIVES_DEEPNETS1M
from .utils import named_layered_modules


import sys
sys.setrecursionlimit(10000)  # for large models like efficientnet_v2_l

t_long = torch.long


class GraphBatch:
    r"""
    Container for a batch of Graph objects.

    Example:

        batch = GraphBatch([Graph(torchvision.models.resnet50())])

    """

    def __init__(self, graphs, dense=False):
        r"""
        :param graphs: iterable, where each item is a Graph object.
        :param dense: create dense node and adjacency matrices (e.g. for transformer)
        """
        self.n_nodes, self.node_feat, self.node_info, self.edges, self.net_args, self.net_inds = [], [], [], [], [], []
        self._n_edges = []
        self.graphs = graphs
        self.dense = dense
        if self.dense:
            self.mask = []

        if graphs is not None:
            if not isinstance(graphs, (list, tuple)):
                graphs = [graphs]
            for graph in graphs:
                self.append(graph)

    def append(self, graph):
        graph_offset = len(self.n_nodes)                    # current index of the graph in a batch
        self.n_nodes.append(len(graph.node_feat))           # number of nodes

        if self.dense:
            self.node_feat.append(graph.node_feat)
            self.edges.append(graph._Adj)
        else:
            self._n_edges.append(len(graph.edges))              # number of edges
            self.node_feat.append(torch.cat((graph.node_feat,   # primitive type
                                             graph_offset + torch.zeros(len(graph.node_feat), 1, dtype=torch.long)),
                                            dim=1))             # graph index for each node
            self.edges.append(torch.cat((graph.edges,
                                         graph_offset + torch.zeros(len(graph.edges), 1, dtype=torch.long)),
                                        dim=1))                 # graph index for each edge

        self.node_info.append(graph.node_info)      # op names, ids, etc.
        self.net_args.append(graph.net_args)        # a dictionary of arguments to construct a Network object
        self.net_inds.append(graph.net_idx)         # network integer identifier (optional)

    def scatter(self, device_ids, nets):
        """
        Distributes the batch of graphs and networks to multiple CUDA devices.
        :param device_ids: list of CUDA devices
        :param nets: list of networks
        :return: list of tuples of networks and corresponding graphs
        """
        n_graphs = len(self.n_nodes)  # number of graphs in a batch
        gpd = int(np.ceil(n_graphs / len(device_ids)))  # number of graphs per device

        if len(device_ids) > 1:
            sorted_idx = self._sort_by_nodes(len(device_ids), gpd)
            nets = [nets[i] for i in sorted_idx]

        chunks_iter = np.arange(0, n_graphs, gpd)
        n_nodes_chunks = [len(self.n_nodes[i:i + gpd]) for i in chunks_iter]
        if self.dense:
            if not isinstance(self.n_nodes, torch.Tensor):
                self.n_nodes = torch.tensor(self.n_nodes, dtype=t_long)
            self.n_nodes = _scatter.apply(device_ids, n_nodes_chunks, 0, self.n_nodes)
        else:
            node_chunks = [sum(self.n_nodes[i:i + gpd]) for i in chunks_iter]
            edge_chunks = [sum(self._n_edges[i:i + gpd]) for i in chunks_iter]
            self._cat()
            self.node_feat = _scatter.apply(device_ids, node_chunks, 0, self.node_feat)
            self.edges = _scatter.apply(device_ids, edge_chunks, 0, self.edges)
            self.n_nodes = _scatter.apply(device_ids, n_nodes_chunks, 0, self.n_nodes)

        batch_lst = []  # each item in the list is a GraphBatch instance
        for device, i in enumerate(chunks_iter):
            # update graph_offset for each device
            graphs = GraphBatch([], dense=self.dense)
            graphs.n_nodes = self.n_nodes[device]

            if self.dense:
                max_nodes = max(graphs.n_nodes)
                graphs.node_feat = [None] * gpd
                graphs.edges = [None] * gpd
                graphs.mask = torch.zeros(gpd, max_nodes, 1, dtype=torch.bool, device=device)
                for k, j in enumerate(range(i, i + gpd)):
                    n = graphs.n_nodes[k]

                    assert n == len(self.node_feat[j]) == len(self.edges[j]), \
                        (i, j, k, n, len(self.node_feat[j]), len(self.edges[j]))

                    graphs.node_feat[k] = F.pad(self.node_feat[j], (0, 0, 0, max_nodes - n), mode='constant')
                    graphs.edges[k] = F.pad(self.edges[j], (0, max_nodes - n, 0, max_nodes - n), mode='constant')
                    graphs.mask[k, :n] = 1

                graphs.node_feat = torch.stack(graphs.node_feat, dim=0).to(device)
                graphs.edges = torch.stack(graphs.edges, dim=0).to(device)

            else:
                self.node_feat[device][:, -1] = self.node_feat[device][:, -1] - gpd * device
                self.edges[device][:, -1] = self.edges[device][:, -1] - gpd * device
                graphs.node_feat = self.node_feat[device]
                graphs.edges = self.edges[device]

            graphs.node_info = self.node_info[i:i + gpd]
            graphs.net_args = self.net_args[i:i + gpd]
            graphs.net_inds = self.net_inds[i:i + gpd]
            batch_lst.append((nets[i:i + gpd], graphs))  # match signature of the GHN forward pass

        return batch_lst

    def to_device(self, device):
        if isinstance(device, (tuple, list)):
            device = device[0]

        if self.on_device(device):
            print('WARNING: GraphBatch is already on device %s.' % str(device))

        self._cat(device)
        self.node_feat = self.node_feat.to(device, non_blocking=True)
        self.edges = self.edges.to(device, non_blocking=True)
        return self

    def on_device(self, device):
        if isinstance(device, (tuple, list)):
            device = device[0]
        return isinstance(self.n_nodes, torch.Tensor) and self.node_feat.device == device

    def to_dense(self, x=None):
        if x is None:
            x = self.node_feat
        B, M, C = len(self.n_nodes), max(self.n_nodes), x.shape[-1]
        node_feat = torch.zeros(B, M, C, device=x.device)
        offset = [0]
        for b in range(B):
            node_feat[b, :self.n_nodes[b]] = x[offset[-1]: offset[-1] + self.n_nodes[b]]
            offset.append(offset[-1] + self.n_nodes[b])
        return node_feat, offset

    def to_sparse(self, x):
        node_feat = torch.cat([x[b, :self.n_nodes[b]] for b in range(len(self.n_nodes))])
        return node_feat

    def _sort_by_nodes(self, num_devices, gpd):
        """
        Sorts graphs and associated attributes in a batch by the number of nodes such
        that the memory consumption is more balanced across GPUs.
        :param num_devices: number of GPU devices (must be more than 1)
        :param gpd: number of graphs per GPU
                                (all GPUs are assumed to receive the same number of graphs)
        :return: indices of sorted graphs
        """
        n_nodes = np.array(self.n_nodes)
        sorted_idx = np.argsort(n_nodes)[::-1]  # decreasing order
        n_nodes = n_nodes[sorted_idx]

        heap = [(0, idx) for idx in range(num_devices)]
        heapq.heapify(heap)
        idx_groups = {}
        for i in range(num_devices):
            idx_groups[i] = []

        for idx, n in enumerate(n_nodes):
            while True:
                set_sum, set_idx = heapq.heappop(heap)
                if len(idx_groups[set_idx]) < gpd:
                    break
            idx_groups[set_idx].append(sorted_idx[idx])
            heapq.heappush(heap, (set_sum + n, set_idx))

        idx = np.concatenate([np.array(v) for v in idx_groups.values()])
        idx = idx[::-1]  # to make fewer nodes on the first device (which is using more)

        # Sort everything according to the idx order
        self.n_nodes = [self.n_nodes[i] for i in idx]
        self._n_edges = [self._n_edges[i] for i in idx]
        self.node_info = [self.node_info[i] for i in idx]
        self.net_args = [self.net_args[i] for i in idx]
        self.net_inds = [self.net_inds[i] for i in idx]
        if self.dense:
            self.node_feat = [self.node_feat[i] for i in idx]
            self.edges = [self.edges[i] for i in idx]
            if len(self.mask) > 0:
                self.mask = [self.mask[i] for i in idx]
        else:
            # update graph_offset for each graph
            node_feat, edges = [], []
            for graph_offset, i in enumerate(idx):
                node_feat_i = self.node_feat[i]
                edges_i = self.edges[i]
                node_feat_i[:, -1] = graph_offset
                edges_i[:, -1] = graph_offset
                node_feat.append(node_feat_i)
                edges.append(edges_i)
            self.node_feat = node_feat
            self.edges = edges

        return idx

    def _cat(self, device='cpu'):
        if not isinstance(self.n_nodes, torch.Tensor):
            self.n_nodes = torch.tensor(self.n_nodes, dtype=t_long, device=device)
        else:
            self.n_nodes = self.n_nodes.to(device, non_blocking=True)

        max_nodes = max(self.n_nodes)

        if not isinstance(self.node_feat, torch.Tensor):

            if self.dense:
                self.mask = torch.zeros(len(self.n_nodes), max_nodes, 1, dtype=torch.bool, device=device)
                for i, x in enumerate(self.node_feat):
                    self.node_feat[i] = F.pad(x, (0, 0, 0, max_nodes - len(x)), mode='constant')
                    self.mask[i, :len(x)] = 1
                    assert self.n_nodes[i] == len(x), (self.n_nodes[i], len(x))
                self.node_feat = torch.stack(self.node_feat, dim=0)
            else:
                self.node_feat = torch.cat(self.node_feat)

        if not isinstance(self.edges, torch.Tensor):
            if self.dense:
                for i, x in enumerate(self.edges):
                    self.edges[i] = F.pad(x, (0, max_nodes - len(x), 0, max_nodes - len(x)), mode='constant')
                self.edges = torch.stack(self.edges, dim=0)
            else:
                self.edges = torch.cat(self.edges)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.n_nodes)

    def __iter__(self):
        for graph in self.graphs:
            yield graph


class Graph:
    r"""
    Container for a computational graph of a neural network.

    Example:

        graph = Graph(torchvision.models.resnet50())

    """

    def __init__(self, model=None, node_feat=None, node_info=None, A=None, edges=None, net_args=None, net_idx=None,
                 ve_cutoff=50, list_all_nodes=False, reduce_graph=True, fix_weight_edges=True, fix_softmax_edges=True,
                 dense=False, verbose=True):
        r"""
        Pass either model or node/edge arguments.
        :param model: Neural Network inherited from nn.Module
        :param node_feat: node features (optional, only if model is None)
        :param node_info: node meta-information (optional, only if model is None)
        :param A: adjacency matrix in the dense format (optional, only if model is None)
        :param edges: adjacency matrix in the sparse format (optional, only if model is None)
        :param net_args: network arguments (optional, only if model is None)
        :param net_idx: network index in the DeepNets-1M dataset (optional, only if model is None)
        :param ve_cutoff: virtual edge cutoff
        :param list_all_nodes: for dataset generation
        :param reduce_graph: remove redundant/unsupported nodes
        :param fix_weight_edges: rewire edges to/from the weight nodes to make it a correct DAG
        :param fix_softmax_edges: rewire edges to/from the softmax nodes to make it consistent with DeepNets-1M DAGs
        :param verbose: print warnings
        """

        assert node_feat is None or model is None, 'either model or other arguments must be specified'

        self.model = model
        self._list_all_nodes = list_all_nodes  # True in case of dataset generation
        self._verbose = verbose
        self._reduce_graph = reduce_graph
        self._fix_weight_edges = fix_weight_edges
        self._fix_softmax_edges = fix_softmax_edges
        self.nx_graph = None  # NetworkX DiGraph instance

        if model is not None:
            # by default assume that the models are for ImageNet images of size 224x224
            sz = model.expected_input_sz if hasattr(model, 'expected_input_sz') else (
                299 if isinstance(model, torchvision.models.Inception3) else 224)
            self.expected_input_sz = sz if isinstance(sz, (tuple, list)) else (3, sz, sz)
            self.n_cells = self.model._n_cells if hasattr(self.model, '_n_cells') else 1
            self._build_graph()   # automatically construct an initial computational graph
            self._add_virtual_edges(ve_cutoff=ve_cutoff)  # add virtual edges
            self._construct_features()  # initialize torch.Tensor node and edge features
            # self.visualize(figname='graph', with_labels=True, font_size=4)  # for debugging purposes
            if not hasattr(model, '_layered_modules'):
                self.model.__dict__['_layered_modules'] = named_layered_modules(self.model)
            self.layered_modules = self.model._layered_modules
        else:
            self.n_nodes = len(node_feat)
            self.node_feat = node_feat
            self.node_info = node_info

            if dense:
                self._Adj = A
            else:
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
            sz = model.expected_input_sz if hasattr(model, 'expected_input_sz') else (
                299 if isinstance(model, torchvision.models.Inception3) else 224)
            expected_input_sz = sz if isinstance(sz, (tuple, list)) else (3, sz, sz)

        device = list(model.parameters())[0].device  # assume all parameters on the same device
        with torch.enable_grad():
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
        Currently, the function is not written very clearly and so it may be improved.
        """

        param_map = {id(weight): (name, module) for name, (weight, module) in self._named_modules().items()}
        nodes, edges, seen = {}, [], {}

        def get_attr(fn):
            """
            Get extra attributes of a node in a computational graph that can help identify the node.
            :param fn:
            :return:
            """
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
            r"""
            Traverse the computational graph of a neural network in the backward direction starting
            from the output node (var).
            :param fn:
            :return:
            """
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
                                name, module = param_map[id(var)]
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
                    nodes[node_link] = {'param_name': leaf['param_name'],
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

        device = list(self.model.parameters())[0].device  # assume all params are on the same device

        with torch.enable_grad():
            # Get model output (var) and then traverse the graph backward
            if hasattr(self.model, 'get_var'):
                # get_var() can be used for the models in which the input is not a 4d tensor (batch of images)
                var = self.model.get_var()
            else:
                var = self.model(torch.randn(2, *self.expected_input_sz, device=device))

            if not isinstance(var, (tuple, list, dict)):
                var = [var]
            if isinstance(var, dict):
                var = list(var.values())
            for v in var:
                if v is not None:
                    traverse_graph(v.grad_fn)  # populate nodes and edges

        nodes_lookup = {key: i for i, key in enumerate(nodes)}
        nodes = [{'id': key, **nodes[key]} for key in nodes_lookup]
        A = np.zeros((len(nodes), len(nodes)))
        for out_node_id, in_node_id in edges:
            A[nodes_lookup[out_node_id], nodes_lookup[in_node_id]] = 1

        self._Adj = A
        self._nodes = nodes
        if self._reduce_graph:
            A, nodes = self._filter_graph()  # Filter graph first time to remove most of the redundant/unsupported nodes

        if self._fix_weight_edges:
            # The weight tensor is often incorrectly placed as a leaf node with a wrong edge direction.
            # For example, for a two layer network like:
            # self.fc = nn.Sequential(
            #             nn.Linear(in_features, in_features),
            #             nn.ReLU(),
            #             nn.Linear(in_features, out_features),
            #         )
            # the edges can be layer0.bias->layer1.bias and layer1.weight->layer1.bias, so layer1.weight does not have
            # incoming edges and is unreachable if we traverse the graph in the forward direction.
            # The loop below corrects the graph by making the edges like layer0.bias->layer1.weight->layer1.bias.

            pattern = 'weight'  # we assume the leaf modules should have a weight attribute
            for i, node in enumerate(nodes):
                if A[:, i].sum() > 0:  # if there are incoming edges to the node, then it already should be correct
                    continue
                if node['param_name'].find(pattern) < 0:  # if no 'weight' string in the name, assume graph is correct
                    continue

                for out_neigh in np.where(A[i, :])[ 0]:  # all nodes with an edge from the weight node, e.g. bias

                    is_same_layer = node['module'] == nodes[out_neigh]['module']

                    if is_same_layer:

                        n_out = len(np.where(A[i, :])[0])  # number of out neighbors the weight node has

                        in_out = np.setdiff1d(np.where(A[:, out_neigh])[0], i)  # incoming to the bias except the i node
                        if len(in_out) == 0:  # if the w (i) is the only incoming to the b, then it should be correct
                            continue

                        nodes[i], nodes[out_neigh] = nodes[out_neigh], nodes[i]
                        A[i, out_neigh], A[out_neigh, i] = 0, 1

                        if n_out == 1:
                            # out_neigh is the weight node after swapping, while i is the bias node
                            out_new = np.setdiff1d(np.where(A[out_neigh, :])[0], i)  # outcoming from w except the bias
                            if len(out_new) == 0:
                                continue
                            A[out_neigh, out_new] = 0  # remove the edges from the weight to out_new
                            A[i, out_new] = 1  # add edges from the bias to out_new

        if self._fix_softmax_edges:
            # Fix softmax/msa edges to be consistent with the GHN/DeepNets-1M code
            pattern = 'softmax'
            self.nx_graph = self._nx_graph_from_adj(A=A)
            for i, node in enumerate(nodes):
                if node['param_name'].lower().find(pattern) < 0:
                    continue
                for out_neigh in np.where(A[i, :])[0]:  # all nodes following i (msa/softmax), usually just one node
                    in_out = np.setdiff1d(np.where(A[:, out_neigh])[0], i)  # nodes coming to out_neigh, except from i
                    for j in in_out:
                        # remove all edges coming to the node next to msa
                        n_paths = 0
                        for _ in nx.all_simple_paths(self.nx_graph, j, out_neigh):
                            n_paths += 1
                            if n_paths > 1:
                                break

                        A[j, out_neigh] = 0  # For ViTs, there should be 2 paths, so remove the 2nd edge to msa/softmax
                        if n_paths == 1:
                            # if only one path from j to out_neigh, then the edge (j, i) will replace (j, out_neigh)
                            A[j, i] = 1

        if sum(A[np.diag_indices_from(A)]) > 0 and self._verbose:
            print('WARNING: diagonal elements of the adjacency matrix should be zero', sum(A[np.diag_indices_from(A)]))

        if self.model is not None and isinstance(self.model, models.SwinTransformer):
            # For SwinTransformer some edges do not match the code, so fixing them manually
            for i, node in enumerate(nodes):
                if node['param_name'].lower().endswith('norm.weight'):
                    for out_neigh in np.where(A[i, :])[0]:
                        if nodes[out_neigh]['param_name'].endswith('norm1.weight') or \
                                nodes[out_neigh]['param_name'].find('Add') >= 0:
                            A[i, out_neigh] = 0
                            target_node = node['param_name'].replace('norm', 'reduction')
                            for j, node2 in enumerate(nodes):
                                if node2['param_name'].find(target_node) >= 0:
                                    A[i, j] = 1
                                    break
                elif node['param_name'].lower().endswith('attn.proj.bias'):
                    for out_neigh in np.where(A[i, :])[0]:
                        if nodes[out_neigh]['param_name'].endswith('reduction.weight'):
                            A[i, out_neigh] = 0  # from attn.bias to reduction.weight
                            for out_neigh2 in np.where(A[out_neigh, :])[0]:
                                if nodes[out_neigh2]['param_name'].startswith('AddBackward'):
                                    A[i, out_neigh2] = 1  # from attn.bias to reduction.weight

        if self._reduce_graph:
            # Filter the graph one more time, since above manipulations could lead to redundant add/concat nodes
            A, nodes = self._filter_graph(unsupported_modules=['Add', 'Cat'])

        # Add input node
        try:
            A = np.pad(A, ((0, 1), (0, 1)), mode='constant')
            nodes.append({'id': 'input', 'param_name': 'input', 'attrs': None, 'module': None})
            # Should work for multiple inputs
            for ind in np.where(A.sum(0) == 0)[0]:  # nodes that do not have any incoming edges
                if nodes[ind]['param_name'].find('weight') >= 0:
                    A[-1, ind] = 1
        except Exception as e:
            print('WARNING: adding input node failed:', e)

        # Sort nodes in a topological order consistent with forward propagation
        try:
            A[np.diag_indices_from(A)] = 0
            ind = np.array(list(nx.topological_sort(nx.DiGraph(A))))
            nodes = [nodes[i] for i in ind]
            A = A[ind, :][:, ind]
        except Exception as e:
            print('WARNING: topological sort failed:', e)

        if self.model is not None:
            # Fix some issues with automatically constructing graphs for some models
            if isinstance(self.model, models.VisionTransformer):
                # Adjust PosEnc for PyTorch ViTs to be consistent with the GHN/DeepNets-1M code
                for i, node in enumerate(nodes):
                    if isinstance(node['module'], (ops.PosEnc, models.vision_transformer.Encoder)):
                        nodes.insert(i + 1,
                                     {'id': 'sum_pos_enc', 'param_name': 'AddBackward0', 'attrs': None, 'module': None})
                        A = np.insert(A, i, 0, axis=0)
                        A = np.insert(A, i, 0, axis=1)
                        A[i, i + 1] = 1  # pos_enc to sum

            elif isinstance(self.model, models.SqueezeNet):
                # Adjust classifier in PyTorch SqueezeNet so that global avg is before classifier
                assert nodes[-1]['param_name'].startswith('MeanBackward'), nodes[-1]
                assert nodes[-3]['param_name'].startswith('classifier'), nodes[-3]
                nodes.insert(len(nodes) - 3, copy.deepcopy(nodes[-1]))
                del nodes[-1]

        self._Adj = A
        self._nodes = nodes

        return

    def _filter_graph(self, unsupported_modules=None):
        r"""
        Remove redundant/unsupported (e.g. internal PyTorch) nodes from the automatically constructed graph.
        This function ended up to be quite messy and potentially brittle, so improvements are welcome.
        :param unsupported_modules: a set/list of unsupported modules
        :return: a tuple of the filtered adjacency matrix and the filtered list of nodes
        """

        # The unsupported_modules will not be added to the graph
        # These are generally some internal PyTorch ops that are not very meaningful (e.g. ViewBackward)
        if unsupported_modules is None:
            unsupported_modules = set()
            for i, node in enumerate(self._nodes):
                ind = node['param_name'].find('Backward')
                op_name = node['param_name'][:len(node['param_name']) if ind == -1 else ind]

                supported = False

                if type(node['module']) in ops.NormLayers and op_name.endswith('.bias'):
                    # In the GHN-2/GHN-3 works the biases of NormLayers are excluded from the graph,
                    # because the biases are always present in NormLayers and thus such nodes are redundant
                    # The parameters of these biases are still predicted

                    pass

                else:
                    for module_name_type in MODULES:
                        if not isinstance(module_name_type, str) and isinstance(node['module'], module_name_type):
                            supported = True
                            break
                if not supported and op_name not in MODULES:
                    unsupported_modules.add(node['param_name'])

            # Add ops requiring extra checks (in the loop below) before removing
            # Staring with 'Mul' to identify CSE nodes (ops like sigmoid/swish need to be in the graph)
            unsupported_modules = ['Mul'] + list(unsupported_modules) + ['Mean', 'Add', 'Cat']

        has_sigmoid_swish_cse = False  # this flag is later used to decide if add a CSE operation or not
        n_incoming = []  # number of incoming edges for each node
        for i, node in enumerate(self._nodes):
            n_incoming.append(len(np.where(self._Adj[:, i])[0]))
            if not has_sigmoid_swish_cse:
                op_name = node['param_name'].lower()
                if op_name.find('sigmoid') >= 0 or op_name.find('swish') >= 0:
                    # Here we make a (quite weak) assumption that networks with the sigmoid/swish ops have CSE nodes
                    has_sigmoid_swish_cse = True

        # Loop over all unsupported_modules and all nodes
        for module_name in unsupported_modules:

            ind_keep = []

            for i, node in enumerate(self._nodes):

                keep = True  # keep the node in the graph
                op_name, attrs = node['param_name'], node['attrs']

                if op_name.find(module_name) >= 0:

                    # Checks for the CSE operation and Add/Concat redundant nodes
                    try:
                        neighbors = dict([(j, self._nodes[i + j]['param_name'].lower()) for j in [-1, -2, 1]])
                        # Check that this node belongs to the classification head by assuming a certain order of nodes
                        classifier_head = np.any([neighbors[j].startswith(('classifier', 'fc', 'head')) for j in [-1, -2]])
                    except Exception as e:
                        print(e, i, len(self._nodes), op_name)
                        classifier_head = True  # tricky case (set to True for now)

                    if op_name.startswith('Mean'):
                        if has_sigmoid_swish_cse:
                            # Do not add the Mean op in CSE unless it's the last global pooling/classification head
                            keep = classifier_head

                    elif op_name.startswith('Mul'):
                        # If below is True, then this Mul op is assumed to be the CSE op
                        # Otherwise, it is some other Mul op that we do not need to add to the graph
                        keep = has_sigmoid_swish_cse and \
                               not classifier_head and \
                               (neighbors[-2].startswith(('hard', 'sigmoid')) or
                                neighbors[1].startswith(('hard', 'sigmoid', 'relu')))

                    elif op_name.startswith(('Cat', 'Add')):  # Concat and Residual (Sum) ops
                        keep = n_incoming[i] > 1  # keep only if > 1 edges are incoming, otherwise the node is redundant
                    else:
                        keep = False

                    if not keep:
                        # If the node is removed, then rewire edges from/to the to-be-removed node to its neighbors
                        for n1 in np.where(self._Adj[i, :])[0]:
                            for n2 in np.where(self._Adj[:, i])[0]:
                                if n1 != n2:
                                    self._Adj[n2, n1] = 1

                if keep:
                    ind_keep.append(i)

            ind_keep = np.array(ind_keep)

            if len(ind_keep) < self._Adj.shape[0]:
                # Remove nodes and edges
                self._Adj = self._Adj[:, ind_keep][ind_keep, :]
                self._nodes = [self._nodes[i] for i in ind_keep]
                n_incoming = [n_incoming[i] for i in ind_keep]

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

        if self._verbose:
            length = nx.shortest_path(self.nx_graph, target=self.n_nodes - 1)
            for node in range(self.n_nodes):
                if node not in length and not self._nodes[node]['param_name'].lower().startswith('aux'):
                    print('WARNING: node={}-{} does not have a path to node={}-{}'.format(
                        node, self._nodes[node]['param_name'], len(self._nodes) - 1, self._nodes[-1]['param_name']))

            # Check that all nodes have a path to the input
            length = nx.shortest_path(self.nx_graph, source=0)
            for node in range(self.n_nodes):
                if node in length:
                    continue
                source_name = self._nodes[0]['param_name']
                target_name = self._nodes[node]['param_name']
                if not (target_name.startswith('pos_enc') or
                        target_name.find('pos_emb') >= 0 or
                        target_name.find('position_bias') >= 0 or
                        source_name.find('position_bias') >= 0):
                    print('WARNING: node={}-{} does not have a path to node={}-{}'.format(
                        0, self._nodes[0]['param_name'], node, self._nodes[node]['param_name']))

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
                try:
                    name = MODULES[param_name[:len(param_name) if ind == -1 else ind]]
                except KeyError as e:
                    name = 'sum'  # used to display redundant nodes when reduce_graph=False

                n_glob_avg += int(name == 'glob_avg')
                if self.n_cells > 1:
                    # Add cell id to the names of pool layers, so that they are matched with proper modules in Network
                    if param_name.startswith(('MaxPool', 'AvgPool')):
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
                        # Pytorch 1.9+ is required to correctly extract pooling attributes,
                        # otherwise the default pooling size of 3 is used
                        sz = (1, 1, 3, 3)
            elif node['module'] is not None:
                sz = (node['module'].weight if param_name.find('weight') >= 0 else node['module'].bias).shape

            if sz is not None:
                if len(sz) == 3 and sz[0] == 1 and min(sz[1:]) > 1:  # [1, 197, 768] -> [1, 768, 14, 14]
                    # setting a 4d size instead of 3d to be consistent with the DeepNets dataset
                    sz_old = sz
                    s = int(np.floor(sz[1] ** 0.5))
                    sz = (1, sz[2], s, s)
                    if self._verbose:
                        print(f'WARNING: setting a 4d size {sz} instead of 3d {tuple(sz_old)}')
                elif len(sz) == 4 and node_ind == len(self._nodes) - 2 and max(sz[2:]) == 1:
                    sz = sz[:2]

            self._param_shapes.append(sz)
            try:
                self.node_feat[node_ind] = primitives_dict[name]
            except KeyError as e:
                print(f'\nError: Op/layer {name} is not present in PRIMITIVES_DEEPNETS1M={PRIMITIVES_DEEPNETS1M}. '
                      f'You can add it there so that it is included in the graph.\n')
                raise

            if node['module'] is not None or name.find('pool') >= 0 or self._list_all_nodes:
                self.node_info[cell_ind].append(
                    [node_ind,
                     param_name if node['module'] is not None else name,
                     name,
                     sz,
                     node_ind == len(self._nodes) - 2 and param_name.find('.weight') >= 0,
                     node_ind == len(self._nodes) - 1 and param_name.find('.bias') >= 0])

        if n_glob_avg != 1 and self._verbose:
            print(f'WARNING: n_glob_avg should be 1 in most architectures, but is {n_glob_avg} in this architecture.')

        self._Adj = torch.tensor(self._Adj, dtype=t_long)

        ind = torch.nonzero(self._Adj)  # rows, cols
        self.edges = torch.cat((ind, self._Adj[ind[:, 0], ind[:, 1]].view(-1, 1)), dim=1)
        return

    def _named_modules(self):
        r"""
        Helper function to automatically build the graphs.
        :return: dictionary of named modules, where the key is the module_name.parameter_name and
        the value is a tuple of (parameter, module)
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

        n_tensors = len(list(self.model.named_parameters()))
        params = dict(self.model.named_parameters())

        if len(modules) > n_tensors:
            if self._verbose:
                print('WARNING: number of tensors found in all submodules ({}) > number of unique tensors ({}). '
                      'This is fine in some models with tied weights.'.format(len(modules), n_tensors))
                for m in modules:
                    if m not in params:
                        print('\t module {} ({}) not in params'.format(m, modules[m][0].shape))
        else:
            assert len(modules) == n_tensors, (len(modules), n_tensors)

        return modules

    def _nx_graph_from_adj(self, A=None, remove_ve=True):
        """
        Creates NetworkX directed graph instance that is used for visualization, virtual edges and graph statistics.
        :param A: adjacency matrix
        :param remove_ve: remove virtual edges from the graph (e.g. to visualize an original graph without ve)
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

    def visualize(self, node_size=50, figname=None, figsize=None, with_labels=False, vis_legend=False,
                  label_offset=0.001, font_size=10, remove_ve=True, detailed_labels=True, **nx_args):
        r"""
        Shows the graphs/legend as in the paper using matplotlib.
        :param node_size: node size
        :param figname: file name to save the figure in the .pdf and .png formats
        :param figsize: (width, height) for a figure
        :param with_labels: show node labels (operations)
        :param vis_legend: True to only visualize the legend (graph will be ignored)
        :param label_offset: positioning of node labels when vis_legend=True
        :param font_size: font size for node labels, used only when with_labels=True
        :param remove_ve: visualize with or without virtual edges (ve)
        :param detailed_labels: use operation full names as labels, used only when with_labels=True
        :param nx_args: extra visualization arguments passed to nx.draw
        :return:
        """

        import matplotlib.pyplot as plt
        from matplotlib import cm as cm

        self._nx_graph_from_adj(remove_ve=remove_ve)

        # first are conv layers, so that they have a similar color
        primitives_ord = [2, 3, 4, 10, 5, 6, 11, 12, 13, 0, 1, 14, 7, 8, 9]
        assert len(PRIMITIVES_DEEPNETS1M) == len(primitives_ord), 'make sure the lists correspond to each other'

        n_primitives = len(primitives_ord)
        color = lambda c: cm.jet(int(np.round(255 * c / n_primitives)))
        primitive_colors = {PRIMITIVES_DEEPNETS1M[i_org]: color(i_new) for i_new, i_org in enumerate(primitives_ord)}
        # manually adjust some colors for better visualization
        primitive_colors['bias'] = '#%02x%02x%02x' % (255, 0, 255)
        primitive_colors['msa'] = '#%02x%02x%02x' % (10, 10, 10)
        primitive_colors['ln'] = '#%02x%02x%02x' % (255, 255, 0)

        node_groups = {'bn':        {'style': {'edgecolors': 'k',       'linewidths': 1,    'node_shape': 's'}},
                       'conv1':     {'style': {'edgecolors': 'k',       'linewidths': 1,    'node_shape': '^'}},
                       'bias':      {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 'd'}},
                       'pos_enc':   {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 's'}},
                       'ln':        {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 's'}},
                       'max_pool':  {'style': {'edgecolors': 'k',       'linewidths': 1,    'node_shape': 'o'}},
                       'glob_avg':  {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 'o'}},
                       'concat':    {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': '^'}},
                       'input':     {'style': {'edgecolors': 'k',       'linewidths': 1.5,  'node_shape': 's'}},
                       'other':     {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 'o'}}}
        for group in ['glob_avg', 'input', 'max_pool']:
            node_groups[group]['node_size'] = (1.75 if group == 'max_pool' else 2) * node_size

        for group in node_groups:
            node_groups[group]['node_lst'] = []
            if 'node_size' not in node_groups[group]['style']:
                node_groups[group]['style']['node_size'] = node_size

        labels, node_colors = {}, []

        if vis_legend:
            node_feat = torch.cat((torch.tensor([n_primitives]).view(-1, 1),
                                   torch.tensor(primitives_ord)[:, None]))
            param_shapes = [(3, 3, 1, 1)] + [None] * n_primitives
        else:
            node_feat = self.node_feat
            param_shapes = self._param_shapes

        for i, (x, sz) in enumerate(zip(node_feat.view(-1), param_shapes)):

            name = PRIMITIVES_DEEPNETS1M[x] if x < n_primitives else 'conv'

            if detailed_labels:
                labels[i] = self._nodes[i]['param_name'].replace('features', 'f').replace(
                    '.weight', '.w').replace('.bias', '.b')
            else:
                labels[i] = name[:20] if x < n_primitives else 'conv_1x1'
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
                                   **node_group['style'],
                                   **nx_args)
        if with_labels:
            nx.draw_networkx_labels(G, pos_labels, labels, font_size=font_size)

        nx.draw_networkx_edges(G, pos,
                               node_size=node_size,
                               width=0 if vis_legend else 1,
                               arrowsize=10,
                               alpha=0 if vis_legend else 1,
                               edge_color='white' if vis_legend else 'k',
                               arrowstyle='-|>',
                               **nx_args)

        plt.grid(False)
        plt.axis('off')
        if figname is not None:
            plt.savefig(figname + '_%d.pdf' % 0, dpi=fig.dpi)
            plt.savefig(figname + '_%d.png' % 0, dpi=fig.dpi, transparent=True)
        else:
            plt.show()


def get_conv_name(module, op_name):
    if op_name.find('bias') >= 0:
        return 'bias'
    elif isinstance(module, nn.Conv2d) and module.groups > 1:
        return 'dil_conv' if min(module.dilation) > 1 else 'sep_conv'
    return 'conv'


# Supported modules/layers
MODULES = {
            nn.Conv2d: get_conv_name,
            nn.Linear: get_conv_name,  # considered equal to conv1x1
            nn.modules.linear.NonDynamicallyQuantizableLinear: get_conv_name,  # linear layer in PyTorch ViT
            nn.modules.activation.MultiheadAttention: get_conv_name,  # linear layer in PyTorch ViT
            transformers.pytorch_utils.Conv1D: get_conv_name,  # for huggingface layers
            nn.BatchNorm2d: lambda module, op_name: 'bn',
            nn.LayerNorm: lambda module, op_name: 'ln',
            models.convnext.LayerNorm2d: lambda module, op_name: 'ln',  # using a separate op (e.g. ln2) could be better
            # We use pos_enc to denote any kind of embedding, which is not the best option
            # Consider adding separate node types (e.g. 'embed') to differentiate between embedding layers
            ops.PosEnc: lambda module, op_name: 'pos_enc',
            nn.modules.sparse.Embedding: lambda module, op_name: 'pos_enc',
            models.vision_transformer.Encoder: lambda module, op_name: 'pos_enc',  # positional encoding in PyTorch ViTs
            'input': 'input',
            'Mean': 'glob_avg',
            'AdaptiveAvgPool2D': 'glob_avg',
            'MaxPool2DWithIndices': 'max_pool',
            'AvgPool2D': 'avg_pool',
            'Softmax': 'msa',  # multi-head self-attention
            'Mul': 'cse',  # ChannelSELayer
            'Add': 'sum',
            'Cat': 'concat',
            'skip_connect': 'sum',  # used to display redundant nodes when reduce_graph=False

            # Adding non-linearities and other layers to the graph is possible as shown below,
            #  but requires adding them in ppuda.deepnets1m.genotypes.PRIMITIVES_DEEPNETS1M:
            # torchvision.models.swin_transformer.ShiftedWindowAttention: lambda module, op_name: 'pos_enc',
            # torchvision.models.convnext.CNBlock: lambda module, op_name: 'layer_scale',
            # 'Gelu': 'gelu',
            # 'Relu': 'relu',

            # Sometimes, existing primitives can be re-used for new operations as we do for
            # models.convnext.LayerNorm2d, however this may be suboptimal compared to introducing a separate op
}
