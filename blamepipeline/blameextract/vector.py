#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-10 11:25:17
"""Functions for putting examples into torch format."""

import torch


def vectorize(ex, model):
    """Torchify a single example."""
    word_dict = model.word_dict
    # Index words
    sents = [torch.tensor([word_dict[w] for w in s], dtype=torch.long) for s in ex['sents']]
    label = ex['label']
    spos = ex['src_pos']
    tpos = ex['tgt_pos']

    # Maybe return without target
    if 'label' not in ex:
        return spos, tpos, sents
    else:
        return spos, tpos, sents, label


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    if isinstance(batch[0], tuple):
        pred_mode = False
        batch_labels = torch.tensor([ex[-1] for ex in batch], dtype=torch.long)
        batch = [b[:-1] for b in batch]
    else:
        pred_mode = True

    sent_offset = 0
    batch_sents = []
    batch_spos, batch_tpos = [], []

    for spos, tpos, sents in batch:
        spos = [(si + sent_offset, wi) for (si, wi) in spos]
        tpos = [(si + sent_offset, wi) for (si, wi) in tpos]
        batch_sents += sents
        sent_offset += len(sents)
        batch_spos.append(spos)
        batch_tpos.append(tpos)

    max_length = max([s.size(0) for s in batch_sents])
    x = torch.zeros(len(batch_sents), max_length, dtype=torch.long)
    x_mask = torch.ones(len(batch_sents), max_length, dtype=torch.uint8)

    for i, s in enumerate(batch_sents):
        x[i, :s.size(0)].copy_(s)
        x_mask[i, :s.size(0)].fill_(0)

    # Maybe return without targets
    if pred_mode:
        return x, x_mask, batch_spos, batch_tpos
    else:
        return x, x_mask, batch_spos, batch_tpos, batch_labels
