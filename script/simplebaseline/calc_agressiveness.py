#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-09-05 15:59:45
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-09-05 16:10:37

'''
Calculate the aggressiveness of a particular entity.
'''

import os
import json
import argparse
from collections import Counter

from blamepipeline import DATA_DIR


def main(args):
    print(args)
    args.dataset = os.path.join(DATA_DIR, 'datasets', args.dataset)
    args.output = os.path.join(DATA_DIR, 'datasets', args.output)

    aggressiveness = {} 
    with open(args.dataset) as f:
        src_count = Counter()
        tgt_count = Counter() 
        entities = set()
        for line in f:
            article = json.loads(line)
            for pair in article['pairs']:
                src = ' '.join(pair['source'])
                tgt = ' '.join(pair['target'])
                src_count[src] += 1
                tgt_count[tgt] += 1
                entities.add(src)
                entities.add(tgt)
        for e in entities:
            aggressiveness[e] = src_count[e] / (src_count[e] + tgt_count[e])
    with open(args.output, 'w') as f:
        for e in sorted(aggressiveness.keys()):
            f.write(f'{e}:{aggressiveness[e]:.4f}\n')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='balala-energy!')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--dataset', default='dataset.json')
    parser.add_argument('--output', default='aggressiveness.txt')
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
