from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import copy
from collections import defaultdict
import collections
import random
import numpy as np
import math
import torch
from rdkit.Geometry import Point3D
import torch.nn.functional as F

# def wiki_dihedral_torch(pos, atomidx):
# # def wiki_dihedral_torch(p0, p1, p2, p3):
#     """formula from Wikipedia article on "Dihedral angle"; formula was removed
#     from the most recent version of article (no idea why, the article is a
#     mess at the moment) but the formula can be found in at this permalink to
#     an old version of the article:
#     https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors
#     uses 1 sqrt, 3 cross products"""
#     # p0 = p[0]
#     # p1 = p[1]
#     # p2 = p[2]
#     # p3 = p[3]

#     p0, p1, p2, p3 = pos[atomidx[:, 0]], pos[atomidx[:, 1]], pos[atomidx[:, 2]], pos[atomidx[:,3]]

#     b0 = -1.0*(p1 - p0)
#     b1 = p2 - p1
#     b2 = p3 - p2

#     b0xb1 = torch.cross(b0, b1)
#     b1xb2 = torch.cross(b2, b1)

#     b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2)

#     # y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
#     # x = np.dot(b0xb1, b1xb2)

#     y = (b0xb1_x_b1xb2 * b1).sum(dim=-1) / torch.norm(b1, dim=-1)
#     x = (b0xb1 * b1xb2).sum(dim=-1)

#     return torch.rad2deg(torch.atan2(y, x))


# def getAngle_torch(pos, idx):
#     # Calculate angles. 0 to pi
#     # idx: i, j, k
#     pos_ji = pos[idx[:, 0]] - pos[idx[:, 1]]
#     pos_jk = pos[idx[:, 2]] - pos[idx[:, 1]]
#     a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
#     b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
#     angle = torch.atan2(b, a)
#     return torch.rad2deg(angle)

# def getAngle_new(conf, atomidx):
#     i, j, k  = np.array(conf.GetAtomPosition(atomidx[0])), np.array(conf.GetAtomPosition(atomidx[1])), np.array(conf.GetAtomPosition(atomidx[2]))
#     pos_ji = i - j 
#     pos_jk = k - j
#     a = (pos_ji * pos_jk).sum(axis=-1) # cos_angle * |pos_ji| * |pos_jk|
#     cross_vec = np.cross(pos_ji, pos_jk)
#     b = np.linalg.norm(cross_vec)
#  # sin_angle * |pos_ji| * |pos_jk|
#     angle = np.arctan2(b, a)
#     # NOTE: no sign
#     # zero_mask = (cross_vec == 0) 
#     # if zero_mask.sum() == 0:
#     #     if (cross_vec < 0).sum() >= 2:
#     #         angle *= -1
#     # else: # has zero
#     #     angle *= np.sign(cross_vec[~zero_mask][0])


#     return angle * 57.3

def wiki_dihedral_torch(dihedral_coords): # [batch_size, pos_len, 4, 3]
# def wiki_dihedral_torch(p0, p1, p2, p3):
    """formula from Wikipedia article on "Dihedral angle"; formula was removed
    from the most recent version of article (no idea why, the article is a
    mess at the moment) but the formula can be found in at this permalink to
    an old version of the article:
    https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors
    uses 1 sqrt, 3 cross products"""
    # p0 = p[0]
    # p1 = p[1]
    # p2 = p[2]
    # p3 = p[3]

    # p0, p1, p2, p3 = pos[atomidx[:, 0]], pos[atomidx[:, 1]], pos[atomidx[:, 2]], pos[atomidx[:,3]]
    p0 = dihedral_coords[..., 0, :].squeeze(-2) # [batch_size, pos_len, 3]
    p1 = dihedral_coords[..., 1, :].squeeze(-2) # [batch_size, pos_len, 3]
    p2 = dihedral_coords[..., 2, :].squeeze(-2) # [batch_size, pos_len, 3]
    p3 = dihedral_coords[..., 3, :].squeeze(-2) # [batch_size, pos_len, 3]

    b0 = -1.0*(p1 - p0) # [batch_size, pos_len, 3]
    b1 = p2 - p1 # [batch_size, pos_len, 3]
    b2 = p3 - p2 # [batch_size, pos_len, 3]

    b0xb1 = torch.cross(b0, b1, dim=-1)
    b1xb2 = torch.cross(b2, b1, dim=-1)

    b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2, dim=-1)

    # y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
    # x = np.dot(b0xb1, b1xb2)

    y = (b0xb1_x_b1xb2 * b1).sum(dim=-1) / torch.norm(b1, dim=-1)
    x = (b0xb1 * b1xb2).sum(dim=-1)

    return torch.rad2deg(torch.atan2(y, x))


def getAngle_torch(angle_coords): # [batch_size, pos_len, 3, 3]
    # Calculate angles. 0 to pi
    # idx: i, j, k
    pi = angle_coords[..., 0, :].squeeze(-2)
    pj = angle_coords[..., 1, :].squeeze(-2)
    pk = angle_coords[..., 2, :].squeeze(-2)
    pos_ji = pi - pj # [batch_size, pos_len, 3]
    pos_jk = pk - pj # [batch_size, pos_len, 3]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|, [batch_size, pos_len]
    b = torch.cross(pos_ji, pos_jk, dim=-1).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return torch.rad2deg(angle)

def getBondLen_torch(edge_coords):
    v1 = edge_coords[..., 0, :]
    v2 = edge_coords[..., 1, :]
    return torch.norm(v1 - v2, p='fro', dim=-1)

def get_coord(pos_idx, coords):
    # pos_idx: [batch_size, pos_len, 2(or 3 or 4)]
    # coords: [batch_size, atom_num, 3]
    # output: [batch_size, pos_len, 2(or 3 or 4), 3]
    pos_len = pos_idx.size(1)
    coords_ext = coords.unsqueeze(1).repeat(1, pos_len, 1, 1) # [batch_size, pos_len, atom_num, 3]
    pos_idx_ext = pos_idx.unsqueeze(-1).repeat(1, 1, 1, 3) # [batch_size, pos_len, 2(or 3 or 4), 3]
    res = torch.gather(input=coords_ext, dim=2, index=pos_idx_ext) # [batch_size, pos_len, 2(or 3 or 4), 3]
    return res

# def make_batch(edge_idx_list, angle_idx_list, dihedral_idx_list, coords_list):
#     # coords_list = [list(coords) for coords in coords_list]
#     for l in [edge_idx_list, angle_idx_list, dihedral_idx_list, coords_list]: # [n, pos_len, 2(or 3 or 4)]
#         len_list = [len(data) for data in l]

def make_batch(data_list): # [n, pos_len, 2(or 3 or 4)]
    len_list = [len(data) for data in data_list]
    max_len = max(len_list)
    data_dim = len(data_list[0][0])
    res = torch.tensor(data_list[0]).new_zeros(len(data_list), max_len, data_dim)
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            torch_data = torch.tensor(data_list[i][j])
            for k in range(len(data_list[i][j])):
                res[i, j, k] = torch_data[k]
    return res

def make_1d_batch(data_list, padding_idx = -10000): # [n, pos_len]
    len_list = [len(data) for data in data_list]
    max_len = max(len_list)
    res = torch.tensor(data_list[0]).new_zeros(len(data_list), max_len) + padding_idx
    for i in range(len(data_list)):
        torch_data = torch.tensor(data_list[i])
        res[i, :len(data_list[i])] = torch_data
    return res

def get_mean_error(res, target, padding_idx=-10000):
    no_valid_mask = target == padding_idx
    res[no_valid_mask] = 0
    target[no_valid_mask] = 0
    return torch.mean(torch.abs(res - target))

def get_gem_loss(decoder_coord, masked_tokens, input_idx, func, target, reduction='mean'):
    # print('input_idx size:', input_idx.size())
    # print('target size:', target.size(), '\n')
    # print('func:', func.__class__, '\n')
    if input_idx.size(1) != target.size(1):
        min_l = min(input_idx.size(1), target.size(1))
        input_idx = input_idx[:, :min_l, :]
        target = target[:, :min_l]

    decoder_coord_masked = decoder_coord.masked_fill(masked_tokens.unsqueeze(-1).expand_as(decoder_coord), -1000000)
    decoder_coord_masked = decoder_coord_masked[:,1:,:]
    
    coords = get_coord(input_idx, decoder_coord_masked) # [batch_size, pos_len, 2(or 3 or 4), 3]
    # if coords.size()
    coords_mask = (coords == -1000000)
    bond_mask = (torch.sum(torch.sum(coords_mask.long(), -1), -1) < 3) # bonds which don't contain noised atoms [batch_size, pos_len]
    real_coords = get_coord(input_idx, decoder_coord[:,1:,:])
    res = func(real_coords.double()).float()
    padding_mask = target == -10000.0
    nan_mask = torch.logical_or(torch.isnan(target), torch.isnan(res))
    mask = ~torch.logical_or(torch.logical_or(bond_mask, padding_mask), nan_mask)
    # print(res[mask])
    # print(target[mask])
    # print(torch.mean(torch.abs(target[mask].float() - res[mask].float())))
    # loss = F.mse_loss(
    #         res[mask],
    #         target[mask],
    #         reduction=reduction,
    #     )
    loss = F.smooth_l1_loss(
            res[mask].float(),
            target[mask].float(),
            reduction=reduction,
            beta=1.0,
        )
    return loss

if __name__ == '__main__':
    from pro_lmdb_dataset import ProLMDBDataset
    path = '/data/kjzheng/datasets/train_smi_tokenized_pos_with_target_small/data.mdb'
    d = ProLMDBDataset(path)
    data_len = 4096
    padding_idx = -10000

    edge_idx_list = []
    angle_idx_list = []
    dihedral_idx_list = []
    coords_list = []
    bond_targets_list = []
    angle_targets_list = []
    dihedral_targets_list = []
    
    for i in range(data_len):
        edge_idx = d[i]['edge_idx']
        angle_idx = d[i]['angle_idx']
        dihedral_idx = d[i]['dihedral_idx']
        coords = d[i]['coordinates'][0]
        bond_targets = d[i]['bond_targets'][0]
        angle_targets = d[i]['angle_targets'][0]
        dihedral_targets = d[i]['dihedral_targets'][0]
        # bond_targets_tensor = torch.tensor(bond_targets)
        # print(bond_targets_tensor)
        # exit()
        edge_idx_list.append(edge_idx)
        angle_idx_list.append(angle_idx)
        dihedral_idx_list.append(dihedral_idx)
        coords_list.append(coords)
        bond_targets_list.append(bond_targets)
        angle_targets_list.append(angle_targets)
        dihedral_targets_list.append(dihedral_targets)
    edge_idx = make_batch(edge_idx_list)
    angle_idx = make_batch(angle_idx_list)
    dihedral_idx = make_batch(dihedral_idx_list)
    coords = make_batch(coords_list)
    bond_targets = make_1d_batch(bond_targets_list, padding_idx = padding_idx) # [n, pos_len]
    angle_targets = make_1d_batch(angle_targets_list, padding_idx = padding_idx) # [n, pos_len]
    dihedral_targets = make_1d_batch(dihedral_targets_list, padding_idx = padding_idx) # [n, pos_len]
    # print(edge_idx.size())
    # print(edge_idx.sum(-1).size())
    # exit()
    # # print(angle_idx.size())
    # # print(dihedral_idx.size())
    # # print(coords.size())
    # edge_coords = get_coord(edge_idx, coords)
    # angle_coords = get_coord(angle_idx, coords)
    # print(angle_coords)
    # print(edge_coords.size())

    # print(dihedral_idx.size())

    edge_coords = get_coord(edge_idx, coords)
    bond_res = getBondLen_torch(edge_coords)
    print(get_mean_error(bond_res, bond_targets, padding_idx=padding_idx))

    angle_coords = get_coord(angle_idx, coords)
    angle_res = getAngle_torch(angle_coords)
    print(get_mean_error(angle_res, angle_targets, padding_idx=padding_idx))

    dihedral_coords = get_coord(dihedral_idx, coords)
    dihedral_res = wiki_dihedral_torch(dihedral_coords)
    print(get_mean_error(dihedral_res, dihedral_targets, padding_idx=padding_idx))




    # print(dihedral_coords.size())
    
    # print(getBondLen_torch(edge_coords))
    # print(bond_targets_list)

    # print(getAngle_torch(angle_coords))
    # print(angle_targets_list)
    # bond_targets_list = make_batch(bond_targets_list)
    # new_data['bond_targets'] = bond_conf_targets
    # new_data['angle_targets'] = angle_conf_targets
    # new_data['dihedral_targets'] = dihedral_conf_targets
