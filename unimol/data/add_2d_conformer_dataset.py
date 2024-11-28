# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from rdkit import Chem
from rdkit.Chem import AllChem


class Add2DConformerDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi, atoms, coordinates):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0
        smi = self.dataset[index][self.smi]
        coordinates_2d = smi2_2Dcoords(smi)
        coordinates = self.dataset[index][self.coordinates]
        coordinates.append(coordinates_2d)
        res = {"smi": smi, "atoms": atoms, "coordinates": coordinates}
        for k in self.dataset[index].keys():
            if k not in res.keys():
                res[k] = self.dataset[index][k]
        if 'bond_targets' in self.dataset[index]:
            res['bond_targets'].append(res['bond_targets'][0])
            res['angle_targets'].append(res['angle_targets'][0])
            res['dihedral_targets'].append(res['dihedral_targets'][0])
            # res['edge_idx'].append(res['edge_idx'][0])
            # res['angle_idx'].append(res['angle_idx'][0])
            # res['dihedral_idx'].append(res['dihedral_idx'][0])
        return res

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(
        coordinates
    ), "2D coordinates shape is not align with {}".format(smi)
    return coordinates
