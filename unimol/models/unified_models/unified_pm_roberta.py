# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicore import utils

from unicore.models import register_model, register_model_architecture
from .fairseq_encoder import FairseqEncoder, FairseqEncoderModel
from .unified_pm_transformer_encoder import UnifiedPMTransformerEncoder

from .layer_norm import LayerNorm
# from fairseq.modules.transformer_sentence_encoder import init_bert_params
from unicore.modules import init_bert_params

logger = logging.getLogger(__name__)


@register_model("unified_pm_roberta")
class UnifiedPMRobertaModel(FairseqEncoderModel):

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        self.token_dropout = args.token_dropout
        self.partially_initialization = args.partially_initialization

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        # self.no_rope = args.no_rope

    # def load_state_dict(self, state_dict, strict=True, **kwargs):
        # super().load_state_dict(state_dict, strict=False, **kwargs)
        # super().load_state_dict(state_dict, strict=True, **kwargs)
    def load_state_dict(self, *args, **kwargs):
        state_dict = args[0]
        my_model_dict = self.state_dict()
        all_key = set(my_model_dict.keys())
        for k, v in list(my_model_dict.items()):
            if k not in state_dict:
                continue
            if self.partially_initialization and ('embed_tokens.weight' in k or 'lm_head.weight' in k):
                esm_embed = state_dict[k]
                v[ : esm_embed.size(0), :] = esm_embed
                my_model_dict[k] = v
                all_key.remove(k)
                continue
            if self.partially_initialization and 'lm_head.bias' in k:
                lm_bias = state_dict[k]
                v[ : lm_bias.size(0)] = lm_bias
                my_model_dict[k] = v
                all_key.remove(k)
                continue
            if v.size() == state_dict[k].size():
                my_model_dict[k] = state_dict[k]
                all_key.remove(k)
        for k in all_key:
            logger.warning(str(k) + ' is not initializated !')
        args = list(args)
        args[0] = my_model_dict
        return super().load_state_dict(*args, **kwargs)

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
            "--max-positions", type=int, help="number of positional embeddings to learn"
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
            "--masked-dist-loss",
            type=float,
            metavar="D",
            default=-1,
            help="weight for masked dist loss",
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            default=1.0,
            help="weight for masked language model loss",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            default=0.01,
            help="weight for x_norm loss",
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
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            default=0.0,
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            default=0.0,
            help="delta encoder pair repr norm loss ratio",
        )

        parser.add_argument(
            "--no-rope",
            action="store_true",
            default=False,
            help="Apply partially initialization",
        )

        parser.add_argument(
            "--no-3d-pe",
            action="store_true",
            default=False,
            help="Apply partially initialization",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        unified_pm_base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            if not safe_hasattr(args, "tokens_per_sample"):
                args.tokens_per_sample = task.max_positions()
            args.max_positions = args.tokens_per_sample

        encoder = UnifiedPMRobertaEncoder(args, task.source_dictionary)

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_edge_type,
        aa_mask=None,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        need_head_weights=False, 
        return_contacts=False,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        x, encoder_distance, encoder_coord, x_norm, delta_encoder_pair_rep_norm = self.encoder(src_tokens, src_distance, src_edge_type, aa_mask=aa_mask, features_only=features_only, \
        return_all_hiddens=return_all_hiddens, token_dropout=self.token_dropout, need_head_weights=need_head_weights, \
        return_contacts=return_contacts, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, encoder_distance, encoder_coord, x_norm, delta_encoder_pair_rep_norm

    def _get_adaptive_head_loss(self):
        norm_loss = 0
        scaling = float(self.args.mha_reg_scale_factor)
        for layer in self.encoder.sentence_encoder.layers:
            norm_loss_layer = 0
            for i in range(layer.self_attn.num_heads):
                start_idx = i * layer.self_attn.head_dim
                end_idx = (i + 1) * layer.self_attn.head_dim
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.q_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.q_proj.bias[start_idx:end_idx])
                    )
                )
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.k_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.k_proj.bias[start_idx:end_idx])
                    )
                )
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.v_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.v_proj.bias[start_idx:end_idx])
                    )
                )

            norm_loss += norm_loss_layer
        return norm_loss

    def _get_adaptive_ffn_loss(self):
        ffn_scale_factor = float(self.args.ffn_reg_scale_factor)
        filter_loss = 0
        for layer in self.encoder.sentence_encoder.layers:
            filter_loss += torch.sum(
                torch.abs(layer.fc1.weight * ffn_scale_factor)
            ) + torch.sum(torch.abs(layer.fc2.weight * ffn_scale_factor))
            filter_loss += torch.sum(
                torch.abs(layer.fc1.bias * ffn_scale_factor)
            ) + torch.sum(torch.abs(layer.fc2.bias * ffn_scale_factor))
        return filter_loss

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

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
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    @property
    def supported_targets(self):
        return {"self"}

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

            # adapt data2vec models
            if (
                "encoder._ema" in state_dict
                and "encoder.lm_head.weight" not in state_dict
            ):
                lm_state = self.encoder.lm_head.state_dict()
                for k, v in lm_state.items():
                    state_dict["encoder.lm_head." + k] = v

            for k in list(state_dict.keys()):
                if k.startswith("encoder.regression_head") or k == "encoder._ema":
                    del state_dict[k]


class RobertaLMHead(nn.Module):
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


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

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

class UnifiedPMRobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        # set any missing default values
        unified_pm_base_architecture(args)
        self.args = args
        self.no_3d_pe = args.no_3d_pe

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)

        masked_token_loss = getattr(args, "masked_token_loss", 0)
        masked_token_loss = max(masked_token_loss, getattr(args, "prot_masked_token_loss", 0) + getattr(args, "mol_masked_token_loss", 0))

        if masked_token_loss > 0:
            self.lm_head = self.build_lm_head(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=(
                    self.sentence_encoder.embed_tokens.weight
                    if not args.untie_weights_roberta
                    else None
                ),
            )
        else:
            self.lm_head = 0

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.masked_dist_loss = getattr(args, "masked_dist_loss", 0)

        self.masked_dist_loss = max(self.masked_dist_loss, getattr(args, "prot_masked_dist_loss", 0) + getattr(args, "mol_masked_dist_loss", 0))

        # print("self.masked_dist_loss:", self.masked_dist_loss)

        if self.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )


    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = UnifiedPMTransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_edge_type,
        aa_mask=None,
        features_only=False,
        return_all_hiddens=False,
        encoder_masked_tokens=None,
        token_dropout=True,
        need_head_weights=False, 
        return_contacts=False,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """

        not_valid_pair_mask = src_edge_type.eq(-1) 
        def get_dist_features(dist, et):
            not_valid_mask = et.eq(-1) # [bsz, n_node, n_node]
            et.masked_fill_(not_valid_mask, 0)
            dist.masked_fill_(not_valid_mask, 0)
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.masked_fill(not_valid_mask.unsqueeze(1).expand_as(graph_attn_bias), 0) # important !
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)

        if self.no_3d_pe:
            graph_attn_bias = graph_attn_bias.masked_fill(torch.ones_like(graph_attn_bias).bool(), 0)

        # print('src_edge_type size:', src_edge_type.size())
        # print('graph_attn_bias size:', graph_attn_bias.size())
        # print('src_tokens size:', src_tokens.size())
        
        x, extra, pair_rep, x_norm = self.extract_features(
            src_tokens, aa_mask=aa_mask, attn_bias=graph_attn_bias, return_all_hiddens=return_all_hiddens, token_dropout=token_dropout,
            need_head_weights=need_head_weights, return_contacts=return_contacts, not_valid_pair_mask=not_valid_pair_mask,
        )
        if features_only:
            encoder_distance = pair_rep
        else:
            if self.masked_dist_loss > 0:
                encoder_distance = self.dist_head(pair_rep)
            else:
                encoder_distance = None
        if not features_only:
            x = self.output_layer(x, masked_tokens=encoder_masked_tokens)
        # return x, extra, encoder_distance, x_norm
        return (
                x,
                encoder_distance,
                None,
                x_norm,
                None,
            )      

    def extract_features(self, src_tokens, aa_mask=None, attn_bias=None, return_all_hiddens=False, token_dropout=True, need_head_weights=False, return_contacts=False,      not_valid_pair_mask=None, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            aa_mask=aa_mask,
            attn_bias=attn_bias,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
            token_dropout=token_dropout,
            not_valid_pair_mask=not_valid_pair_mask,
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None

        return features, {"inner_states": inner_states}, encoder_out["pair_rep"][0], encoder_out["x_norm"][0]

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

def safe_getattr(obj, k, default=None):
    """Returns obj[k] if it exists and is not None, otherwise returns default."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(obj):
        return obj[k] if k in obj and obj[k] is not None else default

    return getattr(obj, k, default)

def safe_hasattr(obj, k):
    """Returns True if the given key exists and is not None."""
    return getattr(obj, k, None) is not None

@register_model_architecture("unified_pm_roberta", "unified_pm_roberta")
def unified_pm_base_architecture(args):
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

    args.no_rope = safe_getattr(args, "no_rope", False)
    args.no_3d_pe = safe_getattr(args, "no_3d_pe", False)


@register_model_architecture("unified_pm_roberta", "unified_pm_roberta_prenorm")
def unified_pm_roberta_prenorm_architecture(args):
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    unified_pm_base_architecture(args)


@register_model_architecture("unified_pm_roberta", "unified_pm_roberta_base")
def unified_pm_roberta_base_architecture(args):
    unified_pm_base_architecture(args)


@register_model_architecture("unified_pm_roberta", "unified_pm_roberta_large")
def unified_pm_roberta_large_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    unified_pm_base_architecture(args)


@register_model_architecture("unified_pm_roberta", "unified_pm_xlm")
def unified_pm_xlm_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 1280 * 4)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    unified_pm_base_architecture(args)

@register_model_architecture("unified_pm_roberta", "unified_pm_roberta_8M")
def unified_pm_roberta_8M_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 320)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 1280)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 20)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    unified_pm_base_architecture(args)

@register_model_architecture("unified_pm_roberta", "unified_pm_roberta_35M")
def unified_pm_roberta_35M_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 480)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 1920)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 20)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    unified_pm_base_architecture(args)

@register_model_architecture("unified_pm_roberta", "unified_pm_roberta_150M")
def unified_pm_roberta_150M_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 30)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 640)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 2560)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 20)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    unified_pm_base_architecture(args)

@register_model_architecture("unified_pm_roberta", "unified_pm_roberta_650M")
def unified_pm_roberta_650M_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 33)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 5120)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 20)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    unified_pm_base_architecture(args)