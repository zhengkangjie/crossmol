# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various fairseq models.
"""

import logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from unicore.models import BaseUnicoreModel

logger = logging.getLogger(__name__)


def check_type(module, expected_type):
    if hasattr(module, "unwrapped_module"):
        assert isinstance(
            module.unwrapped_module, expected_type
        ), f"{type(module.unwrapped_module)} != {expected_type}"
    else:
        assert isinstance(module, expected_type), f"{type(module)} != {expected_type}"


class BaseFairseqModel(BaseUnicoreModel):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()

    @classmethod
    def add_args(cls, parser):
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_args: Optional[Namespace] = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, "")

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    # def set_num_updates(self, num_updates):
    #     """State from trainer to pass along to model at every update."""
    #     for m in self.modules():
    #         if hasattr(m, "set_num_updates") and m != self:
    #             m.set_num_updates(num_updates)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)
        self.apply(_apply)

    def set_epoch(self, epoch):
        for m in self.modules():
            if hasattr(m, "set_epoch") and m != self:
                m.set_epoch(epoch)

    
