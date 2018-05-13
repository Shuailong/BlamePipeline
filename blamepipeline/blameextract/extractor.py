#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:42:04
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-11 16:54:35
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
        feature_size = out_hidden_size * 2
        if args.include_emb:
            feature_size += 2 * args.embedding_dim
        if args.add_self_attn:
            self.self_attn = layers.SelfAttn(args)
            feature_size += 2 * out_hidden_size
        self.linear = nn.Linear(feature_size, 2)

    def forward(self, x, x_mask, batch_spos, batch_tpos):
        """Inputs:
        x = sentence word indices             [sents * len]
        x_mask = sentence padding mask        [sents * len]
        """
        if self.args.unk_entity:
            # mask the entity position
            for spos, tpos in zip(batch_spos, batch_tpos):
                for pos in spos:
                    x[tuple(pos)] = 0
                    x_mask[tuple(pos)] = 1
                for pos in tpos:
                    x[tuple(pos)] = 0
                    x_mask[tuple(pos)] = 1

        x_emb = self.embedding(x)
        if self.args.dropout_emb > 0:
            x_emb = nn.functional.dropout(x_emb, p=self.args.dropout_emb,
                                          training=self.training)
        sent_hiddens = self.sent_rnn(x_emb, x_mask)

        batch_feats = []
        for batch_i, (spos, tpos) in enumerate(zip(batch_spos, batch_tpos)):
            # find entity hidden representation
            s_hids_mean = torch.stack([sent_hiddens[tuple(pos)] for pos in spos], dim=0).mean(0)
            t_hids_mean = torch.stack([sent_hiddens[tuple(pos)] for pos in tpos], dim=0).mean(0)
            features = [s_hids_mean, t_hids_mean]
            if self.args.include_emb:
                # find embeddings
                s_emb, t_emb = x_emb[tuple(spos[0])], x_emb[tuple(tpos[0])]
                features += [s_emb, t_emb]
            if self.args.add_self_attn:
                s_attn_hid = torch.stack([self.self_attn(sent_hiddens[(si, wi)].unsqueeze(0),
                                                         sent_hiddens[si].unsqueeze(0),
                                                         x_mask[si].unsqueeze(0))[0].squeeze(0)
                                          for (si, wi) in spos], dim=0).mean(0)
                t_attn_hid = torch.stack([self.self_attn(sent_hiddens[(si, wi)].unsqueeze(0),
                                                         sent_hiddens[si].unsqueeze(0),
                                                         x_mask[si].unsqueeze(0))[0].squeeze(0)
                                          for (si, wi) in tpos], dim=0).mean(0)
                features += [s_attn_hid, t_attn_hid]

            batch_feats.append(torch.cat(features, dim=0))

        batch_feats = torch.stack(batch_feats, dim=0)
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
