#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-09-03 15:24:05
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-09-06 17:54:16

'''
Input: answer.txt, annotator.txt, output: F1 score
'''

import os
import argparse
from itertools import permutations
import numpy as np

from blamepipeline import DATA_DIR


def main(args):
    print(args)

    args.gold = os.path.join(DATA_DIR, 'datasets', args.gold)
    args.annotation1 = os.path.join(DATA_DIR, 'datasets', args.annotation1)
    args.annotation2 = os.path.join(DATA_DIR, 'datasets', args.annotation2)

    with open(args.gold) as gold_f, open(args.annotation1) as anno_f1, open(args.annotation2) as anno_f2:
        gold_lines = gold_f.readlines()
        anno_lines1 = anno_f1.readlines()
        anno_lines2 = anno_f2.readlines()
        assert len(gold_lines) == len(anno_lines1)
        assert len(gold_lines) == len(anno_lines2)
        assert len(gold_lines) == 200
        gold_lines = gold_lines[1::2]
        anno_lines1 = anno_lines1[1::2]
        anno_lines2 = anno_lines2[1::2]

        gold_l, zy_l, sl_l = [], [], []

        for gold_line, anno_line1, anno_line2 in zip(gold_lines, anno_lines1, anno_lines2):
            try:
                pairstr = gold_line.strip().split(',')
                annostr1 = anno_line1.strip().split(',')
                annostr2 = anno_line2.strip().split(',')
                pairs = set()
                ents = set()
                annopairs1 = set()
                annopairs2 = set()
                for p in pairstr:
                    s, t = p.strip().split('-')
                    ents.add(int(s))
                    ents.add(int(t))
                    pairs.add((int(s), int(t)))
                for p in annostr1:
                    s, t = p.strip().split('-')
                    annopairs1.add((int(s), int(t)))
                for p in annostr2:
                    s, t = p.strip().split('-')
                    annopairs2.add((int(s), int(t)))
                ents = list(ents)
                if len(ents) > 3:
                    continue
                all_pairs = permutations(ents, 2)
                for pair in all_pairs:
                    if pair in pairs:
                        gold_l.append(1)
                    else:
                        gold_l.append(0)
                    if pair in annopairs1:
                        zy_l.append(1)
                    else:
                        zy_l.append(0)
                    if pair in annopairs2:
                        sl_l.append(1)
                    else:
                        sl_l.append(0)

            except Exception as e:
                print(e)
        N = len(gold_l)
        n = 3
        kappa = 0
        n_0 = []
        n_1 = []
        for i in range(N):
            n_0.append((gold_l[i] == 0) + (zy_l[i] == 0) + (sl_l[i] == 0))
            n_1.append((gold_l[i] == 1) + (zy_l[i] == 1) + (sl_l[i] == 1))
        p0 = sum(n_0) / (N * n)
        p1 = sum(n_1) / (N * n)
        Pi = []
        for i in range(N):
            a = (n_0[i] * (n_0[i] - 1) + n_1[i] * (n_1[i] - 1)) / n / (n - 1)
            Pi.append(a)
        P_bar = sum(Pi) / len(Pi)
        P_e = p0 * p0 + p1 * p1

        print(P_bar)
        kappa = (P_bar - P_e) / (1 - P_e)

        print(f'Fleiss kappa: {kappa:.4f}.')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--gold', default='answer.txt')
    parser.add_argument('--annotation1', default='zy.txt')
    parser.add_argument('--annotation2', default='liangshuailong.txt')
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
