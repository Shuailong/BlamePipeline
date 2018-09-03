#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-07-08 19:56:24
"""Blame Extractor utilities."""

import json
import time
import logging
import random

import torch

from .data import BlameTieDataset
from . import vector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Train/dev split
# ------------------------------------------------------------------------------

def split_loader(train_exs, test_exs, args, model, dev_exs=None):
    train_dataset = BlameTieDataset(train_exs, model)
    train_size = len(train_dataset)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=vector.batchify)

    test_dataset = BlameTieDataset(test_exs, model)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=vector.batchify)

    if dev_exs:
        dev_dataset = BlameTieDataset(dev_exs, model)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=1,
            sampler=dev_sampler,
            num_workers=0,
            collate_fn=vector.batchify)
    else:
        dev_size = int(train_size * 0.1)
        train_dev_idxs = list(range(train_size))
        random.shuffle(train_dev_idxs)
        dev_idxs = train_dev_idxs[-dev_size:]
        train_idxs = train_dev_idxs[:train_size - dev_size]
        dev_sampler = torch.utils.data.sampler.SubsetRandomSampler(dev_idxs)
        dev_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            sampler=dev_sampler,
            num_workers=0,
            collate_fn=vector.batchify)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=vector.batchify)

    return train_loader, dev_loader, test_loader


def split_loader_cv(train_exs, args, model, dev_idxs):
    train_dataset = BlameTieDataset(train_exs, model)
    train_idxs = set(range(len(train_dataset))) - set(dev_idxs)
    train_idxs, dev_idxs = sorted(train_idxs), sorted(dev_idxs)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)
    dev_sampler = torch.utils.data.sampler.SubsetRandomSampler(dev_idxs)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=vector.batchify)
    dev_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        sampler=dev_sampler,
        num_workers=0,
        collate_fn=vector.batchify)
    return train_loader, dev_loader

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def load_data(filename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]

    return examples


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
