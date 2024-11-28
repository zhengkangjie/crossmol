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
    TokenizeDataset,
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
    PlusOneDataset,
)
from unicore.tasks import UnicoreTask, register_task
from fairseq.data import MaskTokensDataset

logger = logging.getLogger(__name__)


@register_task("unimol_smi2struct_2d")
class UniMolStruct2DTask(UnicoreTask):
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
            "--atom-dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--smi-dict-name",
            default="smi_dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--uni-dict-name",
            default="uni_dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )
        parser.add_argument(
            "--share-all-embeddings",
            action="store_true",
            help="share all embeddings",
        )
        # parser.add_argument(
        #     "--max-smi-len",
        #     default=1024,
        #     type=int,
        #     help="max length of smiles ",
        # )

        parser.add_argument(
            "--smi-mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--smi-leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--smi-random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )

    def __init__(self, args, atom_dictionary, smi_dictionary):
        super().__init__(args)
        self.atom_dictionary = atom_dictionary
        self.smi_dictionary = smi_dictionary
        self.seed = args.seed
        # add mask token
        self.atom_mask_idx = self.atom_dictionary.add_symbol("[MASK]", is_special=True)

        if self.smi_dictionary is not None:
            self.smi_mask_idx = self.smi_dictionary.add_symbol("[MASK]", is_special=True)
        else:
            self.smi_mask_idx = self.atom_mask_idx

        if self.smi_dictionary is None:
            self.smi_dictionary = self.atom_dictionary

        # self.smi_dictionary.nspecial = self.smi_dictionary.special_index()
        self.smi_dictionary.nspecial = 4

        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True

    @classmethod
    def setup_task(cls, args, **kwargs):
        smi_dictionary = None
        atom_dictionary = None
        if args.share_all_embeddings:
            atom_dictionary = Dictionary.load(os.path.join(args.data, args.uni_dict_name))
            logger.info("Shared dictionary: {} types".format(len(smi_dictionary)))
        else:
            atom_dictionary = Dictionary.load(os.path.join(args.data, args.atom_dict_name))
            smi_dictionary = Dictionary.load(os.path.join(args.data, args.smi_dict_name))
            logger.info("Smiles dictionary: {} types".format(len(smi_dictionary)))
            logger.info("Atoms dictionary: {} types".format(len(atom_dictionary)))
        return cls(args, atom_dictionary, smi_dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")

        raw_dataset = LMDBDataset(split_path)

        def one_dataset(raw_dataset, coord_seed, mask_seed):
            tokenize_smi = KeyDataset(raw_dataset, "smi_tokenized")
            tgt_tokens_pos = KeyDataset(raw_dataset, "atoms_pos")
            tgt_tokens_pos = PlusOneDataset(tgt_tokens_pos)
            smi_tokens = TokenizeDataset(
                tokenize_smi, self.smi_dictionary, max_seq_len=self.args.max_source_positions
            )

            if self.args.mode =='train':
                raw_dataset = Add2DConformerDataset(
                    raw_dataset, "smi", "atoms", "coordinates"
                )
            smi_dataset = KeyDataset(raw_dataset, "smi")
            
            if self.args.smi_mask_prob > 0:
                smi_src_dataset, smi_tgt_dataset = MaskTokensDataset.apply_mask(
                    smi_tokens,
                    self.smi_dictionary,
                    pad_idx=self.smi_dictionary.pad(),
                    mask_idx=self.smi_mask_idx,
                    seed=self.seed,
                    mask_prob=self.args.smi_mask_prob,
                    leave_unmasked_prob=self.args.smi_leave_unmasked_prob,
                    random_token_prob=self.args.smi_random_token_prob,
                )
            else:
                smi_src_dataset, smi_tgt_dataset = smi_tokens, smi_tokens

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
            token_dataset = TokenizeDataset(
                token_dataset, self.atom_dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            expand_dataset = MaskPointsDataset(
                token_dataset,
                coord_dataset,
                self.atom_dictionary,
                pad_idx=self.atom_dictionary.pad(),
                mask_idx=self.atom_mask_idx,
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

            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.atom_dictionary.bos(), self.atom_dictionary.eos()
            )
            tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.atom_dictionary.pad(), self.atom_dictionary.pad()
            )
            tgt_tokens_pos = PrependAndAppend(tgt_tokens_pos, 0, 0)
            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

            smi_src_dataset = PrependAndAppend(
                smi_src_dataset, self.smi_dictionary.bos(), self.smi_dictionary.eos()
            )
            smi_tgt_dataset = PrependAndAppend(
                smi_tgt_dataset, self.smi_dictionary.pad(), self.smi_dictionary.pad()
            )

            edge_type = EdgeTypeDataset(src_dataset, len(self.atom_dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)
            return {
                "smi_tokens": RightPadDataset(
                    smi_src_dataset,
                    pad_idx=self.smi_dictionary.pad(),
                ),
                "tgt_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.atom_dictionary.pad(),
                ),
                "tgt_coord": RightPadDatasetCoord(
                    encoder_coord_dataset,
                    pad_idx=0,
                ),
                "tgt_distance": RightPadDataset2D(
                    encoder_distance_dataset,
                    pad_idx=0,
                ),
                "tgt_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
                "tgt_tokens_pos": RightPadDataset(
                    tgt_tokens_pos,
                    pad_idx=0,
                ),
            }, {
                "tokens_target": RightPadDataset(
                    tgt_dataset, pad_idx=self.atom_dictionary.pad()
                ),
                "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                "smi_tokens_target": RightPadDataset(
                    smi_tgt_dataset, pad_idx=self.smi_dictionary.pad()
                ),
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
