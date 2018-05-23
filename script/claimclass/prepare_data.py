# coding: utf-8

'''
Generate training samples for end2end blame extraction.
'''

import argparse
import os
import json
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from blamepipeline import DATA_DIR

DATASET = os.path.join(DATA_DIR, 'datasets')
logger = logging.getLogger()


def make_samples(data):
    samples = []
    cases = defaultdict(int)
    for d in tqdm(data, total=len(data)):
        claim_sents = [sent for sent in d['claim'] if len(sent) > 2]
        content_sents = [sent for sent in d['content'] if len(sent) > 2]

        for sent in content_sents:
            if sent in claim_sents:
                cases['exact'] += 1
                label = 1
            else:
                sent_set = set(sent)
                for s in claim_sents:
                    claim_set = set(s)
                    simi = len(sent_set & claim_set) / len(sent_set)
                    if simi > 0.8:
                        label = 1
                        cases['fuzzy'] += 1
                        break
                else:
                    label = 0
            if (sent, label) not in samples:
                samples.append((sent, label))
    ratio = sum((label for _, label in samples)) / len(samples)
    logger.info(
        f'Processed {len(samples)} sentences, {cases["exact"]} exact match, {cases["fuzzy"]} fuzzy match. {ratio*100:.2f}% pos ratio.')

    return samples


def stat(sents):
    def statistics(lens):
        min_len = min(lens)
        avg_len = int(sum(lens) / len(lens))
        max_len = max(lens)
        percentile_50 = int(np.percentile(lens, 50))
        percentile_99 = int(np.percentile(lens, 99))
        return min_len, avg_len, max_len, percentile_50, percentile_99

    sent_lens = [len(sent) for sent in sents]
    min_len, avg_len, max_len, percentile_50, percentile_99 = statistics(sent_lens)
    logger.info(
        f'min/avg/max/median/99p w/s: {min_len}/{avg_len:.0f}/{max_len}/{percentile_50:.0f}/{percentile_99:.0f}')


def main(args):
    dataset_file = os.path.join(DATASET, args.dataset)
    logger.info(f'loading data from {dataset_file}')
    data = []
    with open(dataset_file, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception as e:
                logger.error(f'{e}: {line}')
            else:
                data.append(d)
    logger.info('making samples')
    samples = make_samples(data)
    logger.info('calculate sentence statistics')
    stat([sent for sent, label in samples])
    labels = [label for sent, label in samples]
    pos = sum(labels)
    neg = len(labels) - pos
    logger.info(f'pos: {pos}, neg: {neg}, neg/pos = {neg/pos:.1f}')
    sent_file = os.path.join(DATASET, args.sentences)
    logger.info(f'write samples to {sent_file}')

    with open(sent_file, 'w') as f:
        for sample, label in samples:
            line = json.dumps({
                'label': label,
                'sent': sample
            })
            f.write(line + '\n')


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser(description='Make training data')
    parser.add_argument('--dataset', type=str, help='dataset file',
                        default='dataset.json')
    parser.add_argument('--sentences', type=str, help='sentence file',
                        default='sents.json')
    args = parser.parse_args()
    main(args)
