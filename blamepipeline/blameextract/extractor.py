#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:42:04
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-06-01 18:01:11
"""Implementation of the Blame Extractor Class."""

import random

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

        if args.entity_embs:
            self.ent_embedding = nn.Embedding(args.entity_size,
                                              args.entity_embedding_dim,
                                              padding_idx=0)

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
            if args.bidirectional:
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
                self.sent_rnn = self.RNN_TYPES[args.rnn_type](
                    input_size=args.embedding_dim,
                    hidden_size=args.hidden_size,
                    num_layers=args.layers,
                    dropout=args.dropout_rnn,
                    batch_first=True,
                    bidirectional=args.bidirectional)
                out_hidden_size = args.hidden_size
        else:
            out_hidden_size = args.embedding_dim

        if args.pooling == 'attn':
            self.attw = nn.Linear(out_hidden_size, out_hidden_size // 2)
            self.attw2 = nn.Linear(out_hidden_size // 2, 1, bias=False)
            if args.xavier_init:
                nn.init.xavier_uniform(self.attw.weight)
                nn.init.xavier_uniform(self.attw2.weight)

        out_hidden_size *= 2

        if args.entity_embs:
            out_hidden_size += args.entity_embedding_dim * 2

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

    def forward(self, x, x_mask, ents, batch_spos, batch_tpos, batch_sent_chars):
        """Inputs:
        x = sentence word indices             [sents * len]
        x_mask = sentence padding mask        [sents * len]
        ents: batch x 2
        """
        if self.args.entity_embs:
            e_embs = self.ent_embedding(ents)  # batch x 2 x emb
            e_embs = e_embs.view(e_embs.size(0), -1)
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
            if self.args.bidirectional:
                sent_hiddens = self.sent_rnn(x_emb, x_mask)
            else:
                sent_hiddens, _ = self.sent_rnn(x_emb)
        else:
            sent_hiddens = x_emb

        batch_feats = []

        for batch_i, (spos, tpos) in enumerate(zip(batch_spos, batch_tpos)):
            # find entity hidden representation
            if self.args.pooling == 'mean':
                s_hids_pool = torch.stack([sent_hiddens[tuple(pos)] for pos in spos], dim=0).mean(0)
                t_hids_pool = torch.stack([sent_hiddens[tuple(pos)] for pos in tpos], dim=0).mean(0)
            elif self.args.pooling == 'max':
                s_hids_pool = torch.stack([sent_hiddens[tuple(pos)] for pos in spos], dim=0).max(0)[0]
                t_hids_pool = torch.stack([sent_hiddens[tuple(pos)] for pos in tpos], dim=0).max(0)[0]
            elif self.args.pooling == 'attn':
                s_hids_pool = self._attention_pooling(torch.stack([sent_hiddens[tuple(pos)] for pos in spos], dim=0))
                t_hids_pool = self._attention_pooling(torch.stack([sent_hiddens[tuple(pos)] for pos in tpos], dim=0))
            else:
                # self.args.pooling == 'rand':
                s_hids_pool = sent_hiddens[tuple(random.choice(spos))]
                t_hids_pool = sent_hiddens[tuple(random.choice(tpos))]
            feats = [s_hids_pool, t_hids_pool]
            if self.args.entity_embs:
                feats.append(e_embs[batch_i])
            batch_feats.append(torch.cat(feats, dim=0))

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

    def _attention_pooling(self, var):
        energy = nn.functional.tanh(self.attw(var))
        weights = nn.functional.softmax(self.attw2(energy), dim=0)
        return (weights.expand_as(var) * var).sum(0)


class EntityClassifier(nn.Module):
    def __init__(self, args):
        super(EntityClassifier, self).__init__()
        self.args = args
        self.ent_embedding = nn.Embedding(args.entity_size,
                                          args.entity_embedding_dim,
                                          padding_idx=0)
        assert args.feature_size > 0
        self.condense_feature = nn.Linear(2 * args.entity_embedding_dim, args.feature_size)
        self.linear = nn.Linear(args.feature_size, 2)
        if args.xavier_init:
            nn.init.xavier_uniform(self.condense_feature.weight)
            nn.init.xavier_uniform(self.linear.weight)

    def forward(self, x, x_mask, ents, batch_spos, batch_tpos, batch_sent_chars):
        batch_size = ents.size(0)
        e_embs = self.ent_embedding(ents).view(batch_size, -1)
        condensed_feats = self.condense_feature(e_embs)
        if self.args.dropout_final > 0:
            condensed_feats = nn.functional.dropout(condensed_feats, p=self.args.dropout_final,
                                                    training=self.training)
        score = self.linear(nn.functional.tanh(condensed_feats))
        return score
