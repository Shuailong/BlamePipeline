#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-25 17:17:32
"""Functions for putting examples into torch format."""

from collections import defaultdict

import torch


def vectorize(ex, model, uncased=False):
    """Torchify a single example."""
    word_dict = model.word_dict
    label_dict = model.label_dict
    # Index words
    sentences = [[w.lower() for w in s] for s in ex['sents']] if uncased else ex['sents']
    sents = [[word_dict[w] for w in s] for s in sentences]
    labels = [label_dict[label] for label in ex['labels']]
    entities = ex['entities']
    epos = ex['epos']

    # Maybe return without target
    if 'labels' not in ex:
        return entities, epos, sents
    else:
        return entities, epos, sents, labels


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    if isinstance(batch[0], tuple):
        pred_mode = False
        batch_labels = torch.tensor([label for ex in batch for label in ex[-1]], dtype=torch.long)
        batch = [b[:-1] for b in batch]
    else:
        pred_mode = True

    # collate sentences and calculate sentence distance features
    batch_sents = []
    for _, _, sents in batch:
        for sent in sents:
            if sent not in batch_sents:
                batch_sents.append(sent)
    batch_sents = sorted(batch_sents, key=lambda t: -len(t))

    # relocate the entity positions
    batch_epos = defaultdict(set)
    for _, epos, sents in batch:
        for e in epos:
            batch_epos[e] |= {(batch_sents.index(sents[si]), wi) for si, wi in epos[e]}

    max_length = max([len(s) for s in batch_sents])
    x = torch.zeros(len(batch_sents), max_length, dtype=torch.long)
    x_mask = torch.ones(len(batch_sents), max_length, dtype=torch.uint8)

    for i, s in enumerate(batch_sents):
        x[i, :len(s)].copy_(torch.tensor(s, dtype=torch.long))
        x_mask[i, :len(s)].fill_(0)

    batch_entities = [e for ex in batch for e in ex[0]]
    # Maybe return without targets
    if pred_mode:
        return x, x_mask, batch_entities, batch_epos
    else:
        return x, x_mask, batch_entities, batch_epos, batch_labels
