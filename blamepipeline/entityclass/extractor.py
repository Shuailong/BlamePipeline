#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:42:04
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-25 16:58:36
"""Implementation of the EntityClassifier Class."""

import torch
import torch.nn as nn
from . import layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class LSTMContextClassifier(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(LSTMContextClassifier, self).__init__()
        # Store config
        self.args = args
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        self.sent_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        out_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            out_hidden_size *= args.layers

        if args.feature_size > 0:
            self.condense_feature = nn.Linear(out_hidden_size, args.feature_size)
            self.linear = nn.Linear(args.feature_size, args.label_size)
        else:
            self.linear = nn.Linear(out_hidden_size, args.label_size)

    def forward(self, x, x_mask, batch_entities, batch_epos):
        """Inputs:
        x = sentence word indices             [sents * len]
        x_mask = sentence padding mask        [sents * len]
        """
        if self.args.unk_entity:
            # mask the entity position
            for poss in batch_epos.values():
                for pos in poss:
                    x[tuple(pos)] = 0
                    # x_mask[(pos)] = 1

        x_emb = self.embedding(x)
        if self.args.dropout_emb > 0:
            x_emb = nn.functional.dropout(x_emb, p=self.args.dropout_emb,
                                          training=self.training)

        sent_hiddens = self.sent_rnn(x_emb, x_mask)

        batch_feats = []
        for batch_i, e in enumerate(batch_entities):
            # find entity hidden representation
            e_hids_mean = torch.stack([sent_hiddens[tuple(pos)] for pos in batch_epos[e]], dim=0).mean(0)
            batch_feats.append(e_hids_mean)

        batch_feats = torch.stack(batch_feats, dim=0)
        if self.args.dropout_feature > 0:
            batch_feats = nn.functional.dropout(batch_feats, p=self.args.dropout_feature,
                                                training=self.training)
        if self.args.feature_size > 0:
            condensed_feats = self.condense_feature(batch_feats)
            if self.args.dropout_final > 0:
                condensed_feats = nn.functional.dropout(condensed_feats, p=self.args.dropout_final,
                                                        training=self.training)
            score = self.linear(nn.functional.tanh(condensed_feats))
        else:
            score = self.linear(batch_feats)

        return score
