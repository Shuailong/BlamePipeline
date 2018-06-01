#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-06-02 00:28:59
"""Functions for putting examples into torch format."""

import torch


def vectorize(ex, model, uncased=False):
    """Torchify a single example."""
    word_dict = model.word_dict
    entity_dict = model.entity_dict
    # Index words
    sentences = [[w.lower() for w in s] for s in ex['sents']] if uncased else ex['sents']
    src, tgt = ex['src'], ex['tgt']
    spos, tpos = ex['src_pos'], ex['tgt_pos']
    label = ex['label']

    if model.args.unk_entity:
        # mask the entity position
        for si, wi in spos + tpos:
            sentences[si][wi] = '<NULL>'
    sents = [[word_dict[w] for w in s] for s in sentences]

    src_idx = entity_dict[src]
    tgt_idx = entity_dict[tgt]

    # Maybe return without target
    if 'label' not in ex:
        return src_idx, tgt_idx, spos, tpos, sents, sentences
    else:
        return src_idx, tgt_idx, spos, tpos, sents, sentences, label


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    if isinstance(batch[0], tuple):
        pred_mode = False
        batch_labels = torch.Tensor([ex[-1] for ex in batch]).long()
        batch = [b[:-1] for b in batch]
    else:
        pred_mode = True

    # collate sentences and calculate sentence distance features
    batch_sents = []
    batch_sentences = []
    for _, _, _, _, sents, sentences in batch:
        for sent in sents:
            # if sent not in batch_sents:
            batch_sents.append(sent)
        for sent in sentences:
            # if sent not in batch_sentences:
            batch_sentences.append(sent)

    from allennlp.modules.elmo import batch_to_ids
    batch_sent_chars = batch_to_ids(batch_sentences)
    # batch_sent_chars = None

    # relocate the entity positions
    batch_spos, batch_tpos = [], []
    batch_ents = []
    for src_idx, tgt_idx, spos, tpos, sents, _ in batch:
        spos = [(batch_sents.index(sents[si]), wi) for si, wi in spos]
        tpos = [(batch_sents.index(sents[si]), wi) for si, wi in tpos]
        batch_spos.append(spos)
        batch_tpos.append(tpos)
        batch_ents.append([src_idx, tgt_idx])

    ents = torch.LongTensor(batch_ents)

    max_length = max([len(s) for s in batch_sents])
    x = torch.zeros(len(batch_sents), max_length).long()
    x_mask = torch.ones(len(batch_sents), max_length).byte()

    for i, s in enumerate(batch_sents):
        x[i, :len(s)].copy_(torch.Tensor(s).long())
        x_mask[i, :len(s)].fill_(0)

    # Maybe return without targets
    if pred_mode:
        return x, x_mask, ents, batch_spos, batch_tpos, batch_sent_chars
    else:
        return x, x_mask, ents, batch_spos, batch_tpos, batch_sent_chars, batch_labels
