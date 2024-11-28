# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import algos
from ogb.utils.mol import smiles2graph

@torch.jit.script
def convert_to_single_emb(x, offset :int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

class Add2DFeaDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi):
        self.dataset = dataset
        self.smi = smi
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi = self.dataset[index][self.smi]
        res = self.dataset[index].copy()

        pyg_graph = smiles2graph(smi)
        N = pyg_graph["num_nodes"]
        edge_index = pyg_graph["edge_index"]
        edge_attr = torch.from_numpy(pyg_graph["edge_feat"])
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        # edge feature here
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]
                        ] = convert_to_single_emb(edge_attr) + 1
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())

        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())

        # print(edge_input)

        spatial_pos = torch.from_numpy((shortest_path_result)).long()

        res['edge_input'] = edge_input
        res['spatial_pos'] = spatial_pos
        
        return res

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
