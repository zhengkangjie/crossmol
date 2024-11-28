# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import Dictionary
from .unimol import UniMolModel, base_architecture, NonLinearHead
from unicore.modules import LayerNorm
from .transformer_encoder_with_pair import TransformerEncoderWithPair
import numpy as np
from .unified_models import UnifiedPMRobertaModel

logger = logging.getLogger(__name__)


@register_model("docking_pose_modified")
class DockingPoseModifiedModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        parser.add_argument(
            "--token-dropout",
            action="store_true",
            default=False,
            help="Apply token dropout",
        )
        parser.add_argument(
            "--partially-initialization",
            action="store_true",
            default=False,
            help="Apply partially initialization",
        )
        # args for AdaPruning
        # In short, it adds regularizarion for the multihead attention module and feed forward neural nets
        # For more details, please refer to the paper https://openreview.net/forum?id=_CMSV7FTzGI
        parser.add_argument(
            "--mha-reg-scale-factor",
            type=float,
            metavar="D",
            default=0.0,
            help="scaling factor for regularization term in adptive pruning, recommendation is 0.000375",
        )
        parser.add_argument(
            "--ffn-reg-scale-factor",
            type=float,
            metavar="D",
            default=0.0,
            help="scaling factor for regularization term in adptive pruning, recommendation is 0.000375",
        )
        parser.add_argument(
            "--mha-heads-to-keep",
            type=int,
            metavar="D",
            default=-1,
            help="number of heads to keep in each multi-head attention module, -1 means keeping all heads",
        )
        parser.add_argument(
            "--ffn-blocks-to-remove",
            type=int,
            metavar="D",
            default=-1,
            help="number of feedforward blocks to remove in each transformer layer, -1 means keeping all ffn blocks",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )

    def __init__(self, args, dictionary, unipm_model):
        super().__init__()
        

        self.args = args
        self.unipm_model = unipm_model
        self.sep_idx = dictionary.index("[UNK]")
        self.concat_decoder = TransformerEncoderWithPair(
            encoder_layers=4,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            activation_fn="gelu",
        )
        self.cross_distance_project = NonLinearHead(
            args.encoder_embed_dim * 2 + args.encoder_attention_heads, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            args.encoder_embed_dim + args.encoder_attention_heads, "relu"
        )
        self.padding_idx = dictionary.index("[PAD]")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        unipm = UnifiedPMRobertaModel.build_model(args, task)
        return cls(args, task.dictionary, unipm)

    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        masked_tokens=None,
        features_only=True,
        **kwargs
    ):
        # print('mol_src_tokens size:', mol_src_tokens.size())
        # print('pocket_src_tokens size:', pocket_src_tokens.size())
        mol_sz = mol_src_tokens.size(1)
        pocket_sz = pocket_src_tokens.size(1)
        mol_padding_mask = mol_src_tokens.eq(self.padding_idx)
        pocket_padding_mask = pocket_src_tokens.eq(self.padding_idx)
        bsz = mol_src_tokens.size(0)
        sep_tokens = mol_src_tokens.new_full((bsz, 1), self.sep_idx)
        concat_seq = torch.cat([mol_src_tokens, sep_tokens, pocket_src_tokens], dim=1)
        # print('concat_seq size:', concat_seq.size())
        concat_len = concat_seq.size(1)
        concat_dist = mol_src_distance.new_zeros((bsz, concat_len, concat_len))
        concat_dist[:,:mol_sz,:mol_sz] = mol_src_distance
        concat_dist[:,-pocket_sz:,-pocket_sz:] = pocket_src_distance
        concat_edge_type = mol_src_edge_type.new_zeros((bsz, concat_len, concat_len))
        concat_edge_type[:,:mol_sz,:mol_sz] = mol_src_edge_type
        concat_edge_type[:,-pocket_sz:,-pocket_sz:] = pocket_src_edge_type
        aa_mask = torch.zeros_like(concat_seq)
        aa_mask[:, 0] = 1

        outputs = self.unipm_model(
            concat_seq,
            concat_dist,
            concat_edge_type,
            aa_mask=aa_mask,
            features_only=True,
        )
        # exit()
        concat_rep = outputs[0] 
        concat_attn_bias = outputs[1] # bsz, seq_len, seq_len, hidden
        concat_rep = torch.cat([concat_rep[:,:mol_sz,:], concat_rep[:,-pocket_sz:,:]], dim=1)
        concat_attn_bias = torch.cat([concat_attn_bias[:,:mol_sz,:,:], concat_attn_bias[:,-pocket_sz:,:,:]], dim=1)
        concat_attn_bias = torch.cat([concat_attn_bias[:,:,:mol_sz,:], concat_attn_bias[:,:,-pocket_sz:,:]], dim=2)
        concat_attn_bias = concat_attn_bias.reshape((-1, mol_sz + pocket_sz, mol_sz + pocket_sz))

        concat_mask = torch.cat(
            [mol_padding_mask, pocket_padding_mask], dim=-1
        )

        decoder_rep = concat_rep
        decoder_pair_rep = concat_attn_bias
        for i in range(self.args.recycling):
            decoder_outputs = self.concat_decoder(
                decoder_rep, padding_mask=concat_mask, attn_mask=decoder_pair_rep
            )
            decoder_rep = decoder_outputs[0]
            decoder_pair_rep = decoder_outputs[1]
            if i != (self.args.recycling - 1):
                decoder_pair_rep = decoder_pair_rep.permute(0, 3, 1, 2).reshape(
                    -1, mol_sz + pocket_sz, mol_sz + pocket_sz
                )

        mol_decoder = decoder_rep[:, :mol_sz]
        pocket_decoder = decoder_rep[:, mol_sz:]

        mol_pair_decoder_rep = decoder_pair_rep[:, :mol_sz, :mol_sz, :]
        mol_pocket_pair_decoder_rep = (
            decoder_pair_rep[:, :mol_sz, mol_sz:, :]
            + decoder_pair_rep[:, mol_sz:, :mol_sz, :].transpose(1, 2)
        ) / 2.0
        mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float("-inf")] = 0

        cross_rep = torch.cat(
            [
                mol_pocket_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, pocket_sz, 4*hidden_size]

        cross_distance_predict = (
            F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
        )  # batch, mol_sz, pocket_sz

        holo_encoder_pair_rep = torch.cat(
            [
                mol_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, mol_sz, 3*hidden_size]
        holo_distance_predict = self.holo_distance_project(
            holo_encoder_pair_rep
        )  # batch, mol_sz, mol_sz

        return cross_distance_predict, holo_distance_predict

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float("-inf")] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x

def safe_getattr(obj, k, default=None):
    """Returns obj[k] if it exists and is not None, otherwise returns default."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(obj):
        return obj[k] if k in obj and obj[k] is not None else default

    return getattr(obj, k, default)

def safe_hasattr(obj, k):
    """Returns True if the given key exists and is not None."""
    return getattr(obj, k, None) is not None


@register_model_architecture("docking_pose_modified", "docking_pose_modified")
def unimol_docking_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = safe_getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", False)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")

    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = safe_getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # R4F config
    args.spectral_norm_classification_head = safe_getattr(
        args, "spectral_norm_classification_head", False
    )

    args.masked_dist_loss = safe_getattr(args, "masked_dist_loss", -1.0)
    args.token_dropout = safe_getattr(args, "token_dropout", False)

    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)


@register_model_architecture("docking_pose_modified", "docking_pose_modified_prenorm")
def unimol_docking_architecture_prenorm_architecture(args):
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    unimol_docking_architecture(args)


@register_model_architecture("docking_pose_modified", "docking_pose_modified_base")
def docking_pose_modified_base_architecture(args):
    unimol_docking_architecture(args)


@register_model_architecture("docking_pose_modified", "docking_pose_modified_large")
def docking_pose_modified_large_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    unimol_docking_architecture(args)


@register_model_architecture("docking_pose_modified", "docking_pose_modified_xlm")
def docking_pose_modified_xlm_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 1280 * 4)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    unimol_docking_architecture(args)

@register_model_architecture("docking_pose_modified", "docking_pose_modified_35M")
def docking_pose_modified_35M_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 480)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 1920)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 20)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    unimol_docking_architecture(args)
