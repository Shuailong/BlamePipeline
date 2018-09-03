#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-07-08 21:41:44
"""Functions for putting examples into torch format."""


def vectorize(ex, model):
    """Torchify a single example."""
    # Index words
    sents = ex['sents']
    label = ex['label']
    spos = ex['src_pos']
    tpos = ex['tgt_pos']
    sapos = ex['src_pos_original']
    tapos = ex['tgt_pos_original']

    return spos, sapos, tpos, tapos, sents, label


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    batch_spos, batch_sapos, batch_tpos, batch_tapos, batch_sents, batch_labels = zip(*batch)

    return batch_spos, batch_sapos, batch_tpos, batch_tapos, batch_sents, batch_labels
