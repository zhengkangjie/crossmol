# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
# from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .transformer_decoder_with_pair_with_cross_attn_rotary import TransformerDecoderWithPairWithCrossAttnRotary
from typing import Dict, Any, List

from fairseq.modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .uni_transformer_encoder import UniRobertaEncoder

logger = logging.getLogger(__name__)

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

@register_model("smi2struct_rotary")
class Smi2StructRotaryModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
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
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
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
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
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
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help=
            "masked coord loss ratio",
        )
        parser.add_argument(
            "--masked-smi-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--decoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--decoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--decoder-no-pe", action='store_true', help="Don't apply PE for decoder"
        )

        parser.add_argument(
            "--token-dropout",
            action="store_true",
            default=False,
            help="Apply token dropout",
        )
        parser.add_argument(
            "--use-rope",
            action="store_true",
            help="Use RoPE",
        )
        parser.add_argument(
            "--decoder-use-rope",
            action="store_true",
            help="Use RoPE",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            action="store_true",
            help="encoder_learned_pos",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        parser.add_argument(
            "--max-source-positions", type=int, default=1024, help="number of positional embeddings to learn"
        )

        parser.add_argument(
            "--decoder-origin-pe",
            action="store_true",
            help="use origin pe",
        )
        parser.add_argument(
            "--decoder-masked-3d-pe", action='store_true', help="only masked 3D PE for encoder"
        )
        parser.add_argument(
            "--decoder-masked-coords", action='store_true', help="mask the coords"
        )

    def __init__(self, args, source_dictionary, target_dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.decoder_no_pe = args.decoder_no_pe
        self.share_all_embeddings = args.share_all_embeddings
        self.decoder_use_rope = args.decoder_use_rope

        if self.decoder_no_pe:
            self.decoder_use_rope = False

        self.padding_idx = target_dictionary.pad()
        self.encoder_padding_idx = source_dictionary.pad()
        
        self._num_updates = None
        if args.masked_smi_loss > 0:
            self.encoder = UniRobertaEncoder(args, source_dictionary, no_lm_head=False)
        else:
            self.encoder = UniRobertaEncoder(args, source_dictionary, no_lm_head=True)
        if args.share_all_embeddings:
            self.embed_tokens = self.encoder.sentence_encoder.embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                len(target_dictionary), args.encoder_embed_dim, self.padding_idx
            )
        self.decoder = TransformerDecoderWithPairWithCrossAttnRotary(
            encoder_layers=args.decoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            attention_heads=args.decoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.decoder_delta_pair_repr_norm_loss < 0,
            use_rotary_embeddings=self.decoder_use_rope,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.decoder_embed_dim,
                output_dim=len(target_dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )

        K = 128
        n_edge_type = len(target_dictionary) * len(target_dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.decoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.decoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.decoder_attention_heads, args.activation_fn
            )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)
        self.dictionary = target_dictionary
        
        if not self.decoder_no_pe:
            self.embed_positions = SinusoidalPositionalEmbedding(
                embedding_dim = args.encoder_embed_dim,
                padding_idx = target_dictionary.pad(),
                init_size = args.max_seq_len,
            )

        self.mask_idx = target_dictionary.index("[MASK]")
        self.encoder_attention_heads = args.encoder_attention_heads
        self.decoder_origin_pe = args.decoder_origin_pe

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.smi_dictionary, task.atom_dictionary)

    def forward(
        self,
        smi_tokens,
        tgt_tokens,
        tgt_distance,
        tgt_coord,
        tgt_edge_type,
        encoder_masked_tokens=None,
        decoder_masked_tokens=None,
        tgt_tokens_pos=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        # print('decoder_masked_tokens:',decoder_masked_tokens)
        # exit()

        if classification_head_name is not None:
            features_only = True

        encoder_padding_mask = smi_tokens.eq(self.encoder_padding_idx)
        encoder_logits, encoder_output_embedding = self.encoder(
            smi_tokens, 
            token_dropout=self.args.token_dropout, 
            masked_tokens=encoder_masked_tokens,
            retuen_feature=True
        )

        if self.args.masked_smi_loss <= 0:
            encoder_logits = None
        
        decoder_padding_mask = tgt_tokens.eq(self.padding_idx)
        if not decoder_padding_mask.any():
            decoder_padding_mask = None

        x = self.embed_tokens(tgt_tokens)

        if not self.decoder_no_pe and not self.decoder_use_rope:
            if tgt_tokens_pos is not None and not self.decoder_origin_pe: #[bsz, atoms_num]
                smi_length = (~encoder_padding_mask).long().sum(-1, keepdim=True) - 1
                atoms_num = (~decoder_padding_mask).long().sum(-1, keepdim=True) - 1
                tgt_tokens_pos.scatter_(dim=1, index=atoms_num, src=smi_length)# process [eos]
                smi_pe = self.embed_positions(smi_tokens)
                x_pe = torch.gather(input=smi_pe, dim=1, index=tgt_tokens_pos.unsqueeze(-1).expand_as(x_pe))
            else:
                x_pe = self.embed_positions(tgt_tokens)
            x = x + x_pe

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(tgt_distance, tgt_edge_type)

        if self.args.decoder_masked_3d_pe:
            assert decoder_masked_tokens is not None
            mask_m = decoder_masked_tokens # [bsz, seq_len]
            zero_mask = (mask_m.unsqueeze(-1) + mask_m.unsqueeze(1)).bool()
            n_node = tgt_distance.size(-1)
            bsz = tgt_edge_type.size(0)
            zero_mask_exp = zero_mask.unsqueeze(1).expand_as(graph_attn_bias.view(bsz, -1, n_node, n_node)).reshape(-1, n_node, n_node)
            graph_attn_bias = graph_attn_bias.masked_fill(zero_mask_exp, 0)
            
        decoder_tgt_coord = tgt_coord
        if self.args.decoder_masked_coords:
            assert decoder_masked_tokens is not None
            decoder_tgt_coord = tgt_coord.masked_fill(decoder_masked_tokens.unsqueeze(-1).expand_as(tgt_coord), 0)

        (
            decoder_rep,
            decoder_pair_rep,
            delta_decoder_pair_rep,
            decoder_x_norm,
            delta_decoder_pair_rep_norm,
        ) = self.decoder(x, 
            encoder_out=encoder_output_embedding,
            encoder_padding_mask=encoder_padding_mask,
            padding_mask=decoder_padding_mask, 
            attn_mask=graph_attn_bias
        )

        decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        decoder_distance = None
        decoder_coord = None

        logits = decoder_rep
        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(decoder_rep, decoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                # coords_emb = tgt_coord
                coords_emb = decoder_tgt_coord
                if decoder_padding_mask is not None:
                    atom_num = (torch.sum(1 - decoder_padding_mask.type_as(x), dim=1) - 1).view(
                        -1, 1, 1, 1
                    )
                else:
                    atom_num = tgt_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_decoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                decoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                decoder_distance = self.dist_head(decoder_pair_rep)

        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](decoder_rep)
        if self.args.mode == 'infer':
            return decoder_rep, decoder_pair_rep
        else:
            return (
                logits,
                decoder_distance,
                decoder_coord,
                decoder_x_norm,
                delta_decoder_pair_rep_norm,
                encoder_logits,
            )            

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):

    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


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
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@register_model_architecture("smi2struct_rotary", "smi2struct_rotary")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)

    args.decoder_layers = getattr(args, "decoder_layers", 15)
    args.decoder_embed_dim = args.encoder_embed_dim 
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)

    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.encoder_normalize_before = not args.post_ln
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.masked_smi_loss = getattr(args, "masked_smi_loss", -1.0)
    args.decoder_x_norm_loss = getattr(args, "decoder_x_norm_loss", -1.0)
    args.decoder_delta_pair_repr_norm_loss = getattr(args, "decoder_delta_pair_repr_norm_loss", -1.0)

    args.decoder_masked_3d_pe = getattr(args, "decoder_masked_3d_pe", False)
    # args.encoder_unmasked_tokens_only = getattr(args, "encoder_unmasked_tokens_only", False)
    # args.encoder_apply_pe = getattr(args, "encoder_apply_pe", False)
    # args.feed_pair_rep_to_decoder = getattr(args, "feed_pair_rep_to_decoder", False)
    args.decoder_no_pe = getattr(args, "decoder_no_pe", False)
    # args.share_all_embeddings = getattr(args, "share_all_embeddings", False)

    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.token_dropout = getattr(args, "token_dropout", False)
    args.use_rope = getattr(args, "use_rope", False)
    args.decoder_use_rope = getattr(args, "decoder_use_rope", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_origin_pe = getattr(args, "decoder_origin_pe", False)
    args.decoder_masked_coords = getattr(args, "decoder_masked_coords", False)

    

@register_model_architecture("smi2struct_rotary", "smi2struct_rotary_base")
def unimol_base_architecture(args):
    base_architecture(args)

@register_model_architecture("smi2struct_rotary", "smi2struct_rotary_150M")
def base_150M_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 30)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 640)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2560)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 20)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)