#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-10 17:31:11
"""Functions for putting examples into torch format."""


def vectorize(ex, model):
    """Torchify a single example."""
    # Index words
    sents = ex['sents']
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

    if len(batch[0]) == 4:
        pred_mode = False
        batch_labels = [ex[-1] for ex in batch]
        batch = [b[:-1] for b in batch]
    else:
        pred_mode = True

    batch_spos = [ex[0] for ex in batch]
    batch_tpos = [ex[1] for ex in batch]
    batch_sents = [ex[2] for ex in batch]

    # Maybe return without targets
    if pred_mode:
        return batch_spos, batch_tpos, batch_sents
    else:
        return batch_spos, batch_tpos, batch_sents, batch_labels
