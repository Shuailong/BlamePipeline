#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:42:04
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-31 22:34:07
"""Implementation of the Blame Extractor Class."""

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

        if args.pretrain_file != 'elmo':
            # Word embeddings (+1 for padding)
            self.embedding = nn.Embedding(args.vocab_size,
                                          args.embedding_dim,
                                          padding_idx=0)
        else:
            # args.pretrain_file == 'elmo'
            from allennlp.modules.elmo import Elmo
            self.elmo = Elmo(args.elmo_options_file, args.elmo_weights_file, 2,
                             requires_grad=not args.fix_embeddings,
                             dropout=0)
            self.elmo_linear = nn.Linear(1024, args.embedding_dim, bias=False)
            if args.xavier_init:
                nn.init.xavier_uniform(self.elmo_linear.weight)

        if not args.skip_rnn:
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
            out_hidden_size = 2 * args.hidden_size
            if args.concat_rnn_layers:
                out_hidden_size *= args.layers

        else:
            out_hidden_size = args.embedding_dim

        out_hidden_size *= 2

        if args.feature_size > 0:
            self.condense_feature = nn.Linear(out_hidden_size, args.feature_size)
            self.linear = nn.Linear(args.feature_size, 2)
            if args.xavier_init:
                nn.init.xavier_uniform(self.condense_feature.weight)
                nn.init.xavier_uniform(self.linear.weight)
        else:
            self.linear = nn.Linear(out_hidden_size, 2)
            if args.xavier_init:
                nn.init.xavier_uniform(self.linear.weight)

    def forward(self, x, x_mask, batch_spos, batch_tpos, batch_sent_chars):
        """Inputs:
        x = sentence word indices             [sents * len]
        x_mask = sentence padding mask        [sents * len]
        """

        if self.args.pretrain_file != 'elmo':
            x_emb = self.embedding(x)
        else:
            x_elmo = self.elmo(batch_sent_chars)
            x_emb = x_elmo['elmo_representations'][-1]
            # x_mask = x_elmo['mask']
            # x_mask = 1 - x_mask
            x_emb = self.elmo_linear(x_emb)

        if self.args.dropout_emb > 0:
            x_emb = nn.functional.dropout(x_emb, p=self.args.dropout_emb,
                                          training=self.training)

        if not self.args.skip_rnn:
            sent_hiddens = self.sent_rnn(x_emb, x_mask)
        else:
            sent_hiddens = x_emb

        batch_feats = []

        for batch_i, (spos, tpos) in enumerate(zip(batch_spos, batch_tpos)):
            # find entity hidden representation
            s_hids_mean = torch.stack([sent_hiddens[tuple(pos)] for pos in spos], dim=0).mean(0)
            t_hids_mean = torch.stack([sent_hiddens[tuple(pos)] for pos in tpos], dim=0).mean(0)
            # add sentence distence feature
            batch_feats.append(torch.cat([s_hids_mean, t_hids_mean], dim=0))

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


class BiAttentionClassifier(nn.Module):
    def __init__(self, args):
        super(BiAttentionClassifier, self).__init__()
        self.args = args
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        raise NotImplementedError
