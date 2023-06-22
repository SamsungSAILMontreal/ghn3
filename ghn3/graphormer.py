# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer/Graphormer layers.
The layers have to be defined inside a function to be picklable.
Being picklable is required the transformer modules to be re-used during DDP training of the GHN.

"""


import torch
import torch.nn as nn
import math


def create_transformer(Module, Linear, GELU, ReLU, LayerNorm, Dropout, Identity, Sequential):

    class FeedForward(Module):
        """
        Standard MLP applied after each self-attention in Transformer layers.

        """

        def __init__(self, in_features,
                     hidden_features=None,
                     out_features=None,
                     act_layer=GELU,
                     drop=0):
            super().__init__()

            out_features = out_features or in_features
            hidden_features = hidden_features or in_features

            self.net = Sequential(
                Linear(in_features, hidden_features),
                act_layer(),
                Dropout(drop) if drop > 0 else Identity(),
                Linear(hidden_features, out_features),
                Dropout(drop) if drop > 0 else Identity()
            )

        def forward(self, x):
            return self.net(x)

    class EdgeEmbedding(Module):
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
            pe[0, :] = torch.zeros(hid)
            self.embed = nn.Embedding(max_len, hid)
            self.embed.weight.data = pe

        def forward(self, x):
            return self.embed(x)

    class MultiHeadSelfAttentionEdges(Module):
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
            self.to_qkv = Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = Dropout(attn_drop) if attn_drop > 0 else Identity()

            self.to_out = Sequential(Linear(dim, dim), Dropout(proj_drop) if proj_drop > 0 else Identity())

            if self.edge_dim > 0:
                # assume 255+2 is the maximum shortest path distance in graphs
                self.edge_embed = EdgeEmbedding(dim, max_len=257)
                self.proj_e = Sequential(Linear(edge_dim * dim, dim),
                                         ReLU(),
                                         Linear(dim, num_heads))

        def forward(self, x, edges=None, mask=None):
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

            attn = (q @ k.transpose(-2, -1)) * self.scale

            if edges is not None:
                # attention matrix attn is going to be of shape (B, 8, N, N),
                # so we permute the edges (B, N, N, 8) to (B, 8, N, N) before summing up them with attn
                # this operation results in the edge-aware attention matrix that is used to update node features
                attn = attn + edges.permute(0, 3, 1, 2)

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

    class TransformerLayer(Module):
        """
        Based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

        TransformerLayer and Graphormer are combined into a single Module,
        since they only differ in the way edges are processed.

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
                     mlp_ratio=1,
                     qkv_bias=False,
                     act_layer=GELU,
                     eps=1e-5,
                     return_edges=False,
                     stride=1):
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
            self.stride = stride
            self.edge_dim = edge_dim
            if self.edge_dim > 0:
                self.max_degree = 100
                self.max_input_dist = 1000
            self.ln1 = LayerNorm(dim, eps=eps)
            self.attn = MultiHeadSelfAttentionEdges(dim,
                                                    edge_dim=edge_dim,
                                                    num_heads=num_heads,
                                                    qkv_bias=qkv_bias)
            self.ln2 = LayerNorm(dim, eps=eps)
            self.ff = FeedForward(in_features=dim,
                                  hidden_features=int(dim * mlp_ratio),
                                  act_layer=act_layer)

        def forward(self, x, edges=None, mask=None):

            sz = x.shape
            if len(sz) == 2:
                x = x.unsqueeze(0)
            elif len(sz) == 4:
                x = x.reshape(sz[0], sz[1], -1).permute(0, 2, 1)

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

            if len(sz) == 4:
                x = x.permute(0, 2, 1).view(sz[0], x.shape[2], sz[2], sz[3])  # B,C,H,W
                if self.stride > 1:
                    x = x[:, :, ::self.stride, ::self.stride]

            return (x, edges, mask) if self.return_edges else x

    return locals()
