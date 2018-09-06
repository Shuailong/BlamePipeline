#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-09-03 15:24:05
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-09-06 01:21:51

'''
Input: answer.txt, annotator.txt, output: F1 score
'''

import os
import argparse
from itertools import permutations

from blamepipeline import DATA_DIR


def main(args):
    print(args)

    args.gold = os.path.join(DATA_DIR, 'datasets', args.gold)
    args.annotation = os.path.join(DATA_DIR, 'datasets', args.annotation)

    with open(args.gold) as gold_f, open(args.annotation) as anno_f:
        gold_lines = gold_f.readlines()
        anno_lines = anno_f.readlines()
        assert len(gold_lines) == len(anno_lines)
        assert len(gold_lines) == 200
        gold_lines = gold_lines[1::2]
        anno_lines = anno_lines[1::2]
        TP, TN, FP, FN = 0, 0, 0, 0
        gold_1, gold_0, pred_1, pred_0 = 0, 0, 0, 0
        N = 0

        for gold_line, anno_line in zip(gold_lines, anno_lines):
            try:
                pairstr = gold_line.strip().split(',')
                annostr = anno_line.strip().split(',')
                pairs = set()
                ents = set()
                annopairs = set()
                for p in pairstr:
                    s, t = p.strip().split('-')
                    ents.add(int(s))
                    ents.add(int(t))
                    pairs.add((int(s), int(t)))
                for p in annostr:
                    s, t = p.strip().split('-')
                    annopairs.add((int(s), int(t)))
                ents = list(ents)
                if len(ents) > 3:
                    continue
                all_pairs = permutations(ents, 2)
                for pair in all_pairs:
                    N += 1
                    if pair in pairs and pair in annopairs:
                        TP += 1
                        gold_1 += 1
                        pred_1 += 1
                    elif pair not in pairs and pair not in annopairs:
                        TN += 1
                        gold_0 += 1
                        pred_0 += 1
                    elif pair in pairs and pair not in annopairs:
                        FN += 1
                        gold_1 += 1
                        pred_0 += 1
                    elif pair not in pairs and pair in annopairs:
                        FP += 1
                        gold_0 += 1
                        pred_1 += 1
            except Exception as e:
                print(e)
                print(anno_line)

        print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, N: {N}')
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        acc = (TP + TN) / N
        F1 = 2 * precision * recall / (precision + recall)

        p_o = acc
        p_e = (gold_0 * pred_0 + gold_1 * pred_1) / float(N * N)

        kappa = (p_o - p_e) / (1 - p_e)

        print(f'Precision: {precision*100:.2f}%, recall: {recall*100:.2f}%, accuracy: {acc*100:.2f}%, F1: {F1*100:.2f}%. kappa: {kappa:.4f}.')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--gold', default='answer.txt')
    parser.add_argument('--annotation', default='zy.txt')
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
