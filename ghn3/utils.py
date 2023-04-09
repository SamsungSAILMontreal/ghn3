import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math
import types
import ppuda.deepnets1m as deepnets1m
from ppuda.deepnets1m.ops import PosEnc
from ppuda.ghn.nn import GHN
from huggingface_hub import hf_hub_download
import joblib


def from_pretrained(ghn3_name='ghn3xlm16.pt', device='cpu', **kwargs):

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
    else:
        raise NotImplementedError(ghn3_name)

    ghn = GHN(hid=hid, max_shape=(hid, hid, 16, 16), num_classes=1000,
              weight_norm=True, ve=True, layernorm=True, **kwargs)

    ghn.gnn = SequentialMultipleInOut(*[
        GraphormerLayer(dim=hid,
                        num_heads=heads,
                        edge_dim=2 if layer == 0 else 0,
                        return_edges=layer < layers - 1) for layer in range(layers)])

    ghn.centrality_embed_in = nn.Embedding(ghn.gnn[0].max_degree + 1, hid)
    ghn.centrality_embed_out = nn.Embedding(ghn.gnn[0].max_degree + 1, hid)
    ghn.input_dist_embed = nn.Embedding(ghn.gnn[0].max_input_dist + 1, hid)
    ghn.decoder.forward = types.MethodType(decoder_forward, ghn.decoder)
    ghn._tile_params = types.MethodType(_tile_params, ghn)
    ghn._set_params = types.MethodType(_set_params, ghn)
    ghn._normalize = types.MethodType(_normalize, ghn)
    ghn._map_net_params = types.MethodType(_map_net_params, ghn)
    ghn.shape_enc.forward = types.MethodType(shapeenc_forward, ghn.shape_enc)

    for dec_layer in [0, 2]:
        conv = ghn.decoder.conv[dec_layer]
        ghn.decoder.conv[dec_layer] = torch.nn.Linear(conv.weight.shape[1], conv.weight.shape[0]).to(device)
    dec_layer = 1
    conv = ghn.decoder.class_layer_predictor[dec_layer]
    ghn.decoder.class_layer_predictor[dec_layer] = torch.nn.Linear(conv.weight.shape[1],
                                                                   conv.weight.shape[0]).to(device)

    ghn.load_state_dict(joblib.load(hf_hub_download(repo_id='SamsungSAILMontreal/ghn3', filename=ghn3_name)))
    print('loading GHN-3 with %d parameters is done!' % sum([p.numel() for p in ghn.parameters()]))

    # copy the node embeddings for compatibility with GHN-2 code
    ghn.gnn[0].centrality_embed_in = ghn.centrality_embed_in
    ghn.gnn[0].centrality_embed_out = ghn.centrality_embed_out
    ghn.gnn[0].input_dist_embed = ghn.input_dist_embed

    ghn = ghn.eval()

    return ghn


# Graphormer-based layer blocks

class FeedForward(nn.Module):
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
    def __init__(self, C, max_len=5000):
        super().__init__()
        # Based on positional encoding in original Transformers (Attention is All You Need)
        self.C = C
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, C, 2) * (-math.log(10000.0) / C))
        pe = torch.zeros(max_len, C)  # first dim is batch
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.embed = nn.Embedding(max_len, C)
        self.embed.weight.data = pe

    def forward(self, x):
        return self.embed(x)


class MultiHeadSelfAttentionEdges(nn.Module):
    def __init__(self, dim, edge_dim=1, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
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

            # the embedding layers below will be copied from the parent module
            # self.centrality_embed_in = nn.Embedding(self.max_degree + 1, dim)
            # self.centrality_embed_out = nn.Embedding(self.max_degree + 1, dim)
            # self.input_dist_embed = nn.Embedding(self.max_input_dist + 1, dim)

    def forward(self, x, edges):

        if self.edge_dim > 0:
            edges = self.edge_embed(edges)
            edges = edges.reshape(*edges.shape[:-2], -1)
            edges = self.proj_e(edges)

        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale + edges.permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.to_out(x)

        return x, edges


class SequentialMultipleInOut(nn.Sequential):
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
    def __init__(self, dim, edge_dim=0, num_heads=8, mlp_ratio=4, qkv_bias=False, act_layer=nn.GELU, eps=1e-5,
                 return_edges=True):
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

    def forward(self, x, edges, mask):

        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        B, N, C = x.shape
        assert B == 1, ('this code only supports batch size = 1 for evaluation purposes', x.shape)

        if self.edge_dim > 0:

            if edges.dim() == 2:
                # construct a dense adjacency matrix
                edges_dense = torch.zeros(N, N).to(edges)
                edges_dense[edges[:, 0], edges[:, 1]] = edges[:, 2]
                edges = edges_dense.unsqueeze(0)

            edges_1hop = (edges == 1).long()
            x += self.centrality_embed_in(torch.clip(edges_1hop.sum(1), 0, self.max_degree))
            x += self.centrality_embed_out(torch.clip(edges_1hop.sum(2), 0, self.max_degree))
            x += self.input_dist_embed(torch.clip(edges[:, 0, :], 0, self.max_input_dist))
            edges = torch.stack((edges, edges.permute(0, 2, 1)), dim=-1) + 2  # separate fw,bw edge embeddings

        x_attn, edges = self.attn(self.ln1(x), edges)
        x = x + x_attn
        x = x + self.ff(self.ln2(x))

        return (x, edges, mask) if self.return_edges else x[0]


# Modified functions of the GHNs

def decoder_forward(self, x, max_shape=(1, 1, 1, 1), class_pred=False):

    offset = self.out_shape[2] // 2
    N = x.shape[0]
    x = self.fc(x).view(N, -1, *self.out_shape[2:])
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
            # add class token embedding initialized randomly since GHNs were trained on models without this class token
            w = torch.cat((torch.normal(mean=0, std=0.02, size=(1, 1, w.shape[2]), device=w.device), w), dim=1)
    else:
        s2 = min(t[2], s[2]) if len(s) > 2 else 1
        s3 = min(t[3], s[3]) if len(s) > 3 else 1
        offset = (w.shape[-2] // 2, w.shape[-1] // 2)
        if len(s) > 2:
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
        offset = (w.shape[-2] // 2, w.shape[-1] // 2)
        w = w[:t[0], :t[1],
            offset[0] - t[2] // 2: offset[0] + int(np.ceil(t[2] / 2)),
            offset[1] - t[3] // 2: offset[1] + int(np.ceil(t[3] / 2))]

    return w


def _set_params(self, module, tensor, is_w):
    r"""
    Copies the predicted parameter tensor to the appropriate field of the module object.
    :param module: nn.Module
    :param tensor: predicted tensor
    :param is_w: True if it is a weight, False if it is a bias
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

    if self.training:
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
            return p    # do not normalize positional encoding weights

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
            p = torch.tanh(0.2 * p)         # bias is [-1,1]

    return p


def _map_net_params(self, graphs, nets_torch, sanity_check=False):
    r"""
    Matches the parameters in the models with the nodes in the graph.
    Performs additional steps.
    :param graphs: GraphBatch object
    :param nets_torch: a single neural network of a list
    :param sanity_check:
    :return: mapping, params_map
    """
    mapping = {}
    params_map = {}

    nets_torch = [nets_torch] if type(nets_torch) not in [tuple, list] else nets_torch
    for b, (node_info, net) in enumerate(zip(graphs.node_info, nets_torch)):

        target_modules = net.__dict__['_layered_modules'] if self.training else named_layered_modules(net)

        param_ind = torch.sum(graphs.n_nodes[:b]).item()

        for cell_id in range(len(node_info)):
            for (node_ind, p_, name, sz, last_weight, last_bias) in node_info[cell_id]:

                param_name = p_ if p_.endswith(('.weight', '.bias', 'in_proj_weight', 'in_proj_bias')) else p_ + '.weight'
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
                        assert good, \
                            (cell_id, param_name, name,
                             node_info[cell_id],
                             target_modules[cell_id])
                else:
                    sz = matched[0]['sz']

                    def min_sz(j):
                        # to group predicted shapes and improve parallelization and at the same time not to predict much more than needed
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
                    del target_modules[cell_id][param_name]

            if self.training:
                # Prune redundant ops in Network by setting their params to None to speed up training
                for m in target_modules[cell_id].values():
                    if m['is_w']:
                        m['module'].weight = None
                        if hasattr(m['module'], 'bias') and m['module'].bias is not None:
                            m['module'].bias = None

    return mapping, params_map


def shapeenc_forward(self, x, params_map, predict_class_layers=True):
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
                shape_ind[node_ind, i] = self.channels_lookup[sz[i] if sz[i] in self.channels_lookup else self.channels[-1]]
                if self.debug_level and not self.printed_warning:
                    recognized_sz += int(sz[i] in self.channels_lookup_training)
            else:  # for kernel height/width
                shape_ind[node_ind, i] = self.spatial_lookup[sz[i] if sz[i] in self.spatial_lookup else self.spatial[-1]]
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
