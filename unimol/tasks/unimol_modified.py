# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    RightPadDataset2D,
    FromNumpyDataset,
    RawArrayDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    Add2DConformerDataset,
    LMDBDataset,
    MolTokenizeDataset,
    AllZerosDataset
)
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("unimol_modified")
class UniMolModifiedTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--prot-dict-name",
            default="dict_p.txt",
            help="protein dictionary file",
        )
        parser.add_argument(
            "--mol-dict-name",
            default="dict_m.txt",
            help="molecule dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )

    def __init__(self, args, dictionary_p, dictionary_m):
        super().__init__(args)
        for sym in dictionary_m:
            dictionary_p.add_symbol(sym + '_a')
        self.dictionary = dictionary_p
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary_p.index("[MASK]")
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True

    def max_positions(self):
        return self.args.max_positions
    
    @property
    def source_dictionary(self):
        return self.dictionary

    @classmethod
    def load_proteins_dict(cls, args):
        dictionary = Dictionary(bos="[CLS]",
                                pad="[PAD]",
                                eos="[SEP]",
                                unk="[UNK]",
        )
        dictionary.bos_index = dictionary.add_symbol("[CLS]")
        dictionary.pad_index = dictionary.add_symbol("[PAD]")
        dictionary.eos_index = dictionary.add_symbol("[SEP]")
        dictionary.unk_index = dictionary.add_symbol("[UNK]")
        dictionary.add_from_file(os.path.join(args.data, args.prot_dict_name))
        dictionary.add_symbol("<null_1>")
        dictionary.add_symbol("[MASK]", is_special=True)

        logger.info("Proteins dictionary: {} types".format(len(dictionary)))
        return dictionary
    
    @classmethod
    def load_mols_dict(cls, args):
        mol_dict = []
        with open(os.path.join(args.data, args.mol_dict_name), 'r') as fin:
            for idx, line in enumerate(fin):
                sym = line.strip().split()[0].strip()
                mol_dict.append(sym)
        logger.info("Molecules dictionary: {} types".format(len(mol_dict)))
        return mol_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        # dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        # logger.info("dictionary: {} types".format(len(dictionary)))
        dictionary_p = cls.load_proteins_dict(args)
        dictionary_m = cls.load_mols_dict(args)
        return cls(args, dictionary_p, dictionary_m)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")

        raw_dataset = LMDBDataset(split_path)

        def one_dataset(raw_dataset, coord_seed, mask_seed):
            if self.args.mode =='train':
                raw_dataset = Add2DConformerDataset(
                    raw_dataset, "smi", "atoms", "coordinates"
                )
            smi_dataset = KeyDataset(raw_dataset, "smi")
            dataset = ConformerSampleDataset(
                raw_dataset, coord_seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(raw_dataset, dataset)
            dataset = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            )
            dataset = CroppingDataset(
                dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
            )
            dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
            token_dataset = KeyDataset(dataset, "atoms")
            token_dataset = MolTokenizeDataset(
                token_dataset, self.dictionary, max_seq_len=self.args.max_positions
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            expand_dataset = MaskPointsDataset(
                token_dataset,
                coord_dataset,
                self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.args.noise_type,
                noise=self.args.noise,
                seed=mask_seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
            )

            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
            encoder_target_dataset = KeyDataset(expand_dataset, "targets")
            encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

            aa_mask_dataset = AllZerosDataset(encoder_token_dataset)

            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
            )
            aa_mask_dataset = PrependAndAppend(aa_mask_dataset, 1, 1)
            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)
            return {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
                "aa_mask": RightPadDataset(
                    aa_mask_dataset,
                    pad_idx=1,
                ),
            }, {
                "tokens_target": RightPadDataset(
                    tgt_dataset, pad_idx=self.dictionary.pad()
                ),
                "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                "smi_name": RawArrayDataset(smi_dataset),
            }

        net_input, target = one_dataset(raw_dataset, self.args.seed, self.args.seed)
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model
