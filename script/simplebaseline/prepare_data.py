#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 16:18:51
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-22 16:14:00

import argparse
from collections import defaultdict
import json
import os
from itertools import permutations
# from termcolor import colored

from blamepipeline import DATA_DIR

DATASET = os.path.join(DATA_DIR, 'datasets')


def main(args):
    args.dataset_file = os.path.join(DATASET, args.dataset_file)
    args.samples_file = os.path.join(DATASET, args.samples_file)
    print(args)
    articles, sampeles, pos = 0, 0, 0
    with open(args.dataset_file) as f,\
            open(args.samples_file, 'w') as fout:
        for line in f:
            articles += 1
            d = json.loads(line)
            content = d['content']
            pairs = {(p['source'], p['target']) for p in d['pairs']}
            ents = {e for p in pairs for e in p}
            epos = defaultdict(list)
            for si, s in enumerate(content):
                for wi, w in enumerate(s):
                    if w in ents:
                        epos[w].append((si, wi))
            for e in ents:
                assert e in epos

            for src, tgt in permutations(ents, 2):
                if not args.ignore_direction:
                    label = 1 if (src, tgt) in pairs else 0
                else:
                    label = 1 if (src, tgt) in pairs or (tgt, src) in pairs else 0
                if label == 1:
                    pos += 1
                sent_idxs = {si for si, _ in epos[src] + epos[tgt]}
                sents = [content[si] for si in sent_idxs]

                s_abs_pos = epos[src]
                t_abs_pos = epos[tgt]
                s_pos = [(sents.index(content[si]), wi) for si, wi in s_abs_pos]
                t_pos = [(sents.index(content[si]), wi) for si, wi in t_abs_pos]
                dpos = {'src_pos': s_pos, 'src_abs_pos': s_abs_pos, 'tgt_pos': t_pos,
                        'tgt_abs_pos': t_abs_pos, 'sents': sents, 'label': label}
                fout.write(json.dumps(dpos) + '\n')
                sampeles += 1
    print(f'{articles} articles generate {sampeles} samples.')
    print(f'neg / pos = {(sampeles - pos) / pos:.2f}, pos percentage: {pos / sampeles * 100:.2f}%')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data for blame extractor')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--dataset-file', type=str, default='dataset.json')
    parser.add_argument('--samples-file', type=str, default='samples.json')
    parser.add_argument('--ignore-direction', type='bool', default=False)
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
