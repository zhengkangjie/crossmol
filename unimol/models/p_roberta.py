from fairseq.models.roberta import ESM2RobertaModel as ESM2RobertaModel_FS
from fairseq.models.roberta import ESM2RobertaEncoder
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from typing import Dict, Any, List
from fairseq.utils import safe_getattr, safe_hasattr

logger = logging.getLogger(__name__)

@register_model("p_roberta")
class ESM2RobertaModel(BaseUnicoreModel, ESM2RobertaModel_FS):

    def __init__(self, args, encoder):
        ESM2RobertaModel_FS.__init__(self, args, encoder)
        self.args = args
        self.token_dropout = args.token_dropout

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    # def load_state_dict(self, state_dict, strict=True, **kwargs):
    #     # super().load_state_dict(state_dict, strict=False, **kwargs)
    #     super().load_state_dict(state_dict, strict=False, **kwargs)
        
    # def load_state_dict(self, *args, **kwargs):
    #     state_dict = args[0]
    #     my_model_dict = self.state_dict()
    #     all_key = set(my_model_dict.keys())
    #     for k, v in list(my_model_dict.items()):
    #         if k not in state_dict:
    #             continue
    #         # if self.partially_initialization and ('embed_tokens.weight' in k or 'lm_head.weight' in k):
    #         #     esm_embed = state_dict[k]
    #         #     v[ : esm_embed.size(0), :] = esm_embed
    #         #     my_model_dict[k] = v
    #         #     all_key.remove(k)
    #         #     continue
    #         # if self.partially_initialization and 'lm_head.bias' in k:
    #         #     lm_bias = state_dict[k]
    #         #     v[ : lm_bias.size(0)] = lm_bias
    #         #     my_model_dict[k] = v
    #         #     all_key.remove(k)
    #         #     continue
    #         if v.size() == state_dict[k].size():
    #             my_model_dict[k] = state_dict[k]
    #             all_key.remove(k)
    #     for k in all_key:
    #         logger.warning(str(k) + ' is not initializated !')
    #     args = list(args)
    #     args[0] = my_model_dict
    #     return super().load_state_dict(*args, **kwargs)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        p_base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            if not safe_hasattr(args, "tokens_per_sample"):
                args.tokens_per_sample = task.max_positions()
            args.max_positions = args.tokens_per_sample

        encoder = ESM2RobertaEncoder(args, task.source_dictionary)

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, encoder)
    
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
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
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

def safe_getattr(obj, k, default=None):
    """Returns obj[k] if it exists and is not None, otherwise returns default."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(obj):
        return obj[k] if k in obj and obj[k] is not None else default

    return getattr(obj, k, default)

@register_model_architecture("p_roberta", "p_roberta")
def p_base_architecture(args):
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
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = safe_getattr(
        args, "encoder_normalize_before", False
    )
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = safe_getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = safe_getattr(
        args, "spectral_norm_classification_head", False
    )