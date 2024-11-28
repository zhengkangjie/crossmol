# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import torch
from unicore.data import Dictionary
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import numpy as np

class ListTokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_seq_len: int=512,
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        raw_data = self.dataset[index]
        result = list(raw_data)
        if len(result) >= self.max_seq_len:
            result = result[:self.max_seq_len - 1]
        return result