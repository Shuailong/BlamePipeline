#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-10 16:06:42

"""Data processing/loading helpers."""

import logging

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .vector import vectorize

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# PyTorch dataset class for Blame data.
# ------------------------------------------------------------------------------


class BlameTieDataset(Dataset):
    def __init__(self, examples, model):
        self.model = model
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.model)


# ------------------------------------------------------------------------------
# PyTorch sampler
# ------------------------------------------------------------------------------


class SubsetWeightedRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices, weights, num_samples=None, replacement=True):
        self.indices = indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        sampled = iter(torch.multinomial(self.weights, self.num_samples, self.replacement))
        return (self.indices[i] for i in sampled)

    def __len__(self):
        return self.num_samples
