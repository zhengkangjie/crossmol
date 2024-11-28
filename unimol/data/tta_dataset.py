# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class TTADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, conf_size=10):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        smi = self.dataset[smi_idx]["smi"]
        target = self.dataset[smi_idx]["target"]
        res = {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "smi": smi,
            "target": target,
        }
        if "smi_tokenized" in self.dataset[smi_idx]:
            res["smi_tokenized"] = self.dataset[smi_idx]["smi_tokenized"]

        if 'bond_targets' in self.dataset[smi_idx]:
            res['bond_targets'] = self.dataset[smi_idx]["bond_targets"][coord_idx]
            res['angle_targets'] = self.dataset[smi_idx]["angle_targets"][coord_idx]
            res['dihedral_targets'] = self.dataset[smi_idx]["dihedral_targets"][coord_idx]
            res['dihedral_idx'] = self.dataset[smi_idx]["dihedral_idx"]
            res['angle_idx'] = self.dataset[smi_idx]["angle_idx"]
            res['edge_idx'] = self.dataset[smi_idx]["edge_idx"]
            # edge_idx_dataset = KeyDataset(dataset, "edge_idx")
            # angle_idx_dataset = KeyDataset(dataset, "angle_idx")
            # dihedral_idx_dataset = KeyDataset(dataset, "dihedral_idx")
            if len(res['dihedral_targets']) == 0 or len(res['dihedral_idx']) == 0:
                res['dihedral_targets'] = [-10000.0]
                res['dihedral_idx'] = [[0, 0, 0, 0]]
            if len(res['angle_targets']) == 0 or len(res['angle_idx']) == 0:
                res['angle_targets'] = [-10000.0]
                res['angle_idx'] = [[0, 0, 0]]
            if len(res['bond_targets']) == 0 or len(res['edge_idx']) == 0:
                res['bond_targets'] = [-10000.0]
                res['edge_idx'] = [[0, 0]]
        if 'atoms_pos' in self.dataset[smi_idx]:
            res["atoms_pos"] = self.dataset[smi_idx]["atoms_pos"]
            # if len(res["atoms_pos"]) !=  len(res["atoms"][res["atoms"]!='H']):
            #     print('length error!', len(res["atoms_pos"]), len(res["atoms"][res["atoms"]!='H']))
            #     exit()
        return res

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class TTADockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        holo_coordinates,
        holo_pocket_coordinates,
        is_train=True,
        conf_size=10,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.holo_coordinates = holo_coordinates
        self.holo_pocket_coordinates = holo_pocket_coordinates
        self.is_train = is_train
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        pocket_atoms = np.array(
            [item[0] for item in self.dataset[smi_idx][self.pocket_atoms]]
        )
        pocket_coordinates = np.array(self.dataset[smi_idx][self.pocket_coordinates][0])
        if self.is_train:
            holo_coordinates = np.array(self.dataset[smi_idx][self.holo_coordinates][0])
            holo_pocket_coordinates = np.array(
                self.dataset[smi_idx][self.holo_pocket_coordinates][0]
            )
        else:
            holo_coordinates = coordinates
            holo_pocket_coordinates = pocket_coordinates

        smi = self.dataset[smi_idx]["smi"]
        pocket = self.dataset[smi_idx]["pocket"]

        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_coordinates": holo_coordinates.astype(np.float32),
            "holo_pocket_coordinates": holo_pocket_coordinates.astype(np.float32),
            "smi": smi,
            "pocket": pocket,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
