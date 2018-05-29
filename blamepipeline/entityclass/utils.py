#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-28 14:57:14
"""Blame Extractor utilities."""

import json
import time
import logging
import random
from collections import Counter

import torch

from .data import Dictionary
from .data import BlameTieDataset
from .data import SubsetWeightedRandomSampler
from . import vector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Train/dev split
# ------------------------------------------------------------------------------

def split_loader(train_exs, test_exs, args, model, dev_exs=None):
    train_dataset = BlameTieDataset(train_exs, model)
    train_size = len(train_dataset)
    train_idxs = list(range(train_size))

    test_dataset = BlameTieDataset(test_exs, model)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda)

    if dev_exs:
        dev_dataset = BlameTieDataset(dev_exs, model)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda)
        train_idxs_ = train_idxs
    else:
        dev_size = int(train_size * args.valid_size)
        random.shuffle(train_idxs)
        dev_idxs = train_idxs[-dev_size:]
        dev_exs = [train_exs[i] for i in dev_idxs]
        train_idxs_ = train_idxs[:train_size - dev_size]
        dev_sampler = torch.utils.data.sampler.SubsetRandomSampler(dev_idxs)
        dev_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda)
    train_exs_ = [train_exs[i] for i in train_idxs_]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs_)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda)

    if args.debug:
        # dev and test vocabulary coverage in train
        vocab_coverage(args, model, train_exs_, dev_exs, test_exs)

    return train_loader, dev_loader, test_loader


def split_loader_cv(train_exs, args, model, test_idxs):
    train_dataset = BlameTieDataset(train_exs, model)
    train_idxs = list(set(range(len(train_dataset))) - set(test_idxs))
    random.shuffle(train_idxs)
    train_idxs_ = train_idxs[:int(len(train_idxs) * (1 - args.valid_size))]
    dev_idxs = train_idxs[int(len(train_idxs) * (1 - args.valid_size)):]

    dev_sampler = torch.utils.data.sampler.SubsetRandomSampler(dev_idxs)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idxs)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs_)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda)
    dev_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda)
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda)

    if args.debug:
        # dev and test vocabulary coverage in train
        train_exs_ = [train_exs[i] for i in train_idxs_]
        dev_exs = [train_exs[i] for i in dev_idxs]
        test_exs = [train_exs[i] for i in test_idxs]
        vocab_coverage(args, model, train_exs_, dev_exs, test_exs)

    return train_loader, dev_loader, test_loader


def vocab_coverage(args, model, train_exs, dev_exs, test_exs):
    train_vocab = set(load_words(args, train_exs, cutoff=0))
    dev_vocab = set(load_words(args, dev_exs, cutoff=0))
    test_vocab = set(load_words(args, test_exs, cutoff=0))
    words = set(model.word_dict.tokens())
    dev_coverage = len(dev_vocab & train_vocab & words) / len(dev_vocab)
    test_coverage = len(test_vocab & train_vocab & words) / len(test_vocab)
    logger.debug(f'train/dev/test samples: {len(train_exs)}/{len(dev_exs)}/{len(test_exs)}')
    logger.debug(f'train/dev/test vocab: {len(train_vocab)}/{len(dev_vocab)}/{len(test_vocab)}')
    logger.debug(f'dev vocab coverage: {dev_coverage*100:.2f}%')
    logger.debug(f'test vocab coverage: {test_coverage*100:.2f}%')


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
# Dictionary building
# ------------------------------------------------------------------------------


def load_words(args, examples, cutoff=1):
    """Iterate and index all the words in examples (documents + questions)."""

    words = Counter()
    for ex in examples:
        for s in ex['sents']:
            for w in s:
                if args.uncased:
                    w = w.lower()
                w = Dictionary.normalize(w)
                words[w] += 1
    words = (w for w, f in words.most_common() if f > cutoff)
    return words


def build_word_dict(args, examples, cutoff=1):
    """Return a dictionary from sentence words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples, cutoff=cutoff):
        word_dict.add(w)
    return word_dict


def build_label_dict(args, examples):
    label_dict = set()
    for ex in examples:
        for label in ex['labels']:
            label_dict.add(label)
    label_dict = {label: index for index, label in enumerate(sorted(label_dict))}
    return label_dict


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
