#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-22 22:40:50
"""Functions for putting examples into torch format."""

import torch


def vectorize(ex, model, uncased=False):
    """Torchify a single example."""
    word_dict = model.word_dict
    # Index words
    sentences = [[w.lower() for w in s] for s in ex['sents']] if uncased else ex['sents']
    sents = [[word_dict[w] for w in s] for s in sentences]
    label = ex['label']
    spos = ex['src_pos']
    tpos = ex['tgt_pos']
    spos_ori = ex['src_pos_original']
    tpos_ori = ex['tgt_pos_original']

    # Maybe return without target
    if 'label' not in ex:
        return spos, tpos, spos_ori, tpos_ori, sents
    else:
        return spos, tpos, spos_ori, tpos_ori, sents, label


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    if isinstance(batch[0], tuple):
        pred_mode = False
        batch_labels = torch.tensor([ex[-1] for ex in batch], dtype=torch.long)
        batch = [b[:-1] for b in batch]
    else:
        pred_mode = True

    # collate sentences and calculate sentence distance features
    batch_sents = []
    batch_dist_feats = []
    for _, _, spos_ori, tpos_ori, sents in batch:
        for sent in sents:
            if sent not in batch_sents:
                batch_sents.append(sent)
        batch_dist_feats.append(min((abs(s_i - t_i) for (s_i, _), (t_i, _) in zip(spos_ori, tpos_ori))))
    batch_sents = sorted(batch_sents, key=lambda t: -len(t))

    # relocate the entity positions
    batch_spos, batch_tpos = [], []
    for spos, tpos, _, _, sents in batch:
        spos = [(batch_sents.index(sents[si]), wi) for si, wi in spos]
        tpos = [(batch_sents.index(sents[si]), wi) for si, wi in tpos]
        batch_spos.append(spos)
        batch_tpos.append(tpos)

    max_length = max([len(s) for s in batch_sents])
    x = torch.zeros(len(batch_sents), max_length, dtype=torch.long)
    x_mask = torch.ones(len(batch_sents), max_length, dtype=torch.uint8)

    for i, s in enumerate(batch_sents):
        x[i, :len(s)].copy_(torch.tensor(s, dtype=torch.long))
        x_mask[i, :len(s)].fill_(0)

    # batch_dist_feats = torch.tensor(batch_dist_feats, dtype=torch.float)

    # Maybe return without targets
    if pred_mode:
        return x, x_mask, batch_spos, batch_tpos, batch_dist_feats
    else:
        return x, x_mask, batch_spos, batch_tpos, batch_dist_feats, batch_labels
