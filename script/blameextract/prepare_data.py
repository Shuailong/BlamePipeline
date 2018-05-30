#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 16:18:51
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-30 20:21:13

'''
Within an article, merge the same entities and construct a map from entity name to entity id.
In the blame tie and article content, replace entity names by its id.
'''

import argparse
from collections import defaultdict
import os
import json
import pickle  # json cannot accpet tuple key
from itertools import permutations
from statistics import mean, stdev

from termcolor import colored
from tqdm import tqdm


from blamepipeline import DATA_DIR as DATA_ROOT

DATA_DIR = os.path.join(DATA_ROOT, 'datasets')


def entity_merge_global(args):
    '''
    Merge entities globally.
    entity2alias: {longest entity name(tuple) ->all alias sort by word length in decrease order (tuples)}
    entity2id: {entity name (tuple) -> longest entity name (str) }
    '''
    entity_json_path = os.path.join(DATA_DIR, 'entity.json')
    entity_txt_path = os.path.join(DATA_DIR, 'entity.txt')
    entity2id_path = os.path.join(DATA_DIR, 'entity2id.pkl')

    if not args.data_force and os.path.exists(entity_json_path) and\
            os.path.exists(entity_txt_path) and\
            os.path.exists(entity2id_path):
        print(f'Load entity2id from {entity2id_path}')
        with open(entity2id_path, 'rb') as f:
            dict = pickle.load(f)
        return dict

    # read all entities in dataset
    with open(args.dataset_file) as f:
        all_ents = set()
        for line in f:
            all_ents |= {tuple(e) for e in json.loads(line)['entities']}

    # one step transform
    transform_d = {}
    for e in tqdm(all_ents, total=len(all_ents), desc='find transforms'):
        for e_ in all_ents:
            if len(e) <= 1 or e == e_:
                continue
            if ' '.join(e) in ' '.join(e_):
                transform_d[e] = e_
                e = e_
            elif ' '.join(e_) in ' '.join(e):
                transform_d[e_] = e
            elif len(e) == 2 and len(e_) == 3 and e[0] == e_[0] and e[-1] == e_[-1] and e_[1].endswith('.'):
                # middle name
                transform_d[e] = e_
                print(f'{e} -> {e_}')
            elif len(e) == 3 and len(e_) == 2 and e[0] == e_[0] and e[-1] == e_[-1] and e[1].endswith('.'):
                transform_d[e_] = e
                print(f'{e_} -> {e}')

    # construct dict
    entity2id = {}
    entity2alias = defaultdict(set)
    for e in all_ents:
        e_ = e
        while e_ in transform_d:
            e_ = transform_d[e_]
        entity2id[e] = ' '.join(e_)
        entity2alias[e_].add(e)
    # sort by entity len
    entity2alias = {e: sorted(alias, key=lambda t: -len(t)) for e, alias in entity2alias.items()}
    # save
    with open(entity_json_path, 'w') as entity_j,\
            open(entity_txt_path, 'w') as entity_t:
        for eid, alias in entity2alias.items():
            entity_j.write(json.dumps({'entityid': eid, 'alias': alias}) + '\n')
            entity_t.write(f'{" ".join(eid)}: {", ".join([" ".join(e) for e in alias])}\n')
    print(f'Write entity and alias to {entity_json_path} and {entity_txt_path}')

    with open(entity2id_path, 'wb') as eid_f:
        pickle.dump(entity2id, eid_f)
    print(f'Write entity2id dict to {entity2id_path}')

    return entity2id


def entity_merge_local(entities):
    # one step transform
    transform_d = {}
    for e in entities:
        for e_ in entities:
            if len(e) <= 1 or e == e_:
                continue
            if ' '.join(e) in ' '.join(e_):
                transform_d[e] = e_
                e = e_
            elif ' '.join(e_) in ' '.join(e):
                transform_d[e_] = e
            elif len(e) == 2 and len(e_) == 3 and e[0] == e_[0] and e[-1] == e_[-1] and e_[1].endswith('.'):
                # middle name
                transform_d[e] = e_
            elif len(e) == 3 and len(e_) == 2 and e[0] == e_[0] and e[-1] == e_[-1] and e[1].endswith('.'):
                transform_d[e_] = e
    entity2id = {}
    entity2alias = defaultdict(set)
    for e in entities:
        e_ = e
        while e_ in transform_d:
            e_ = transform_d[e_]
        entity2id[e] = ' '.join(e_)
        entity2alias[e_].add(e)
    return entity2id


def main(args):
    args.samples_file = os.path.join(DATA_DIR, args.samples_file)
    args.dataset_file = os.path.join(DATA_DIR, args.dataset_file)
    fname, ext = os.path.splitext(args.samples_file)
    if args.ignore_direction:
        args.samples_file = fname + '-undirected' + ext
    else:
        args.samples_file = fname + '-directed' + ext
    print(args)

    if args.merge_entity == 'global':
        entity2id = entity_merge_global(args)

    # debug
    num_articles, num_samples, num_pos_samples = 0, 0, 0
    all_entities_mismatch, tie_entities_mismatch = 0, 0
    num_entities_dist, samples_ratio_dist, blame_tie_dist = [], [], []

    samples = []
    with open(args.dataset_file) as f:
        lines = sum((1 for _ in f))
    with open(args.dataset_file) as f:
        for line in tqdm(f, total=lines, desc='generate samples'):
            d = json.loads(line)
            content = d['content']
            pairs = sorted({(tuple(p['source']), tuple(p['target'])) for p in d['pairs']})
            if args.all_entities:
                all_entities = sorted({tuple(e) for e in d['entities']})
            else:
                all_entities = sorted({e for p in pairs for e in p})
            if args.merge_entity == 'local':
                entity2id = entity_merge_local(all_entities)
            pairs_entities_ids = sorted({(entity2id[s], entity2id[t])
                                         for s, t in pairs if entity2id[s] != entity2id[t]})
            all_entities_ids = sorted({entity2id[e] for e in all_entities})

            # merge entity words in *content* into a single token, and replace by id
            # sorted ents by word len to avoid mismatch
            all_entities = sorted(all_entities, key=lambda t: -len(t))
            epos = defaultdict(list)
            for si, s in enumerate(content):
                wi = 0
                while wi < len(s):
                    for e in all_entities:
                        e_list, e_len = list(e), len(e)
                        if s[wi: wi + e_len] == e_list or\
                                s[wi: wi + e_len - 1] == e_list[:-1] and s[wi + e_len - 1] == e_list[-1] + '.':
                            s[wi: wi + e_len] = [entity2id[e]]
                            break
                    wi += 1

            # find positions for entities
            epos = defaultdict(list)
            for si, s in enumerate(content):
                for wi, w in enumerate(s):
                    if w in all_entities_ids:
                        epos[w].append((si, wi))

            for i, e in enumerate(all_entities_ids):
                if e not in epos:
                    all_entities_mismatch += 1
            for p in pairs_entities_ids:
                for e in p:
                    if e not in epos:
                        tie_entities_mismatch += 1

            # # remove entities which cannot be found in article
            all_entities_ids = sorted({e for e in all_entities_ids if e in epos})
            pairs_entities_ids = sorted({(s, t) for s, t in pairs_entities_ids if s in epos and t in epos})
            if len(pairs_entities_ids) == 0:
                continue

            # make negative samples and select sentences
            article_pos_num = 0
            article_samples_num = 0
            article_samples = []
            for src, tgt in permutations(all_entities_ids, 2):
                if not args.ignore_direction:
                    label = 1 if (src, tgt) in pairs_entities_ids else 0
                else:
                    label = 1 if (src, tgt) in pairs_entities_ids or (tgt, src) in pairs_entities_ids else 0
                if label == 1:
                    article_pos_num += 1
                article_samples_num += 1
                sent_idxs = {si for si, _ in epos[src] + epos[tgt]}
                sents = [content[si] for si in sent_idxs]
                s_pos = [(sents.index(content[si]), wi) for si, wi in epos[src]]
                t_pos = [(sents.index(content[si]), wi) for si, wi in epos[tgt]]
                sample = {'src_pos': s_pos, 'tgt_pos': t_pos,
                          'src_pos_original': epos[src], 'tgt_pos_original': epos[tgt],
                          'sents': sents, 'label': label}
                article_samples.append(sample)

                if label == 1:
                    blame_tie_dist.append(min((abs(si - ti) for (si, _), (ti, _) in zip(epos[src], epos[tgt]))))
            assert article_pos_num > 0

            num_pos_samples += article_pos_num
            num_samples += article_samples_num
            num_articles += 1
            num_entities_dist.append(len(all_entities_ids))
            samples_ratio_dist.append((article_samples_num - article_pos_num) / article_pos_num)
            samples.append(article_samples)

    if args.split:
        train_articles_num = int(num_articles * 0.8)
        dev_articles_num = int(num_articles * 0.1)

        train_articles_samples = samples[:train_articles_num]
        dev_articles_samples = samples[train_articles_num:train_articles_num + dev_articles_num]
        test_articles_samples = samples[train_articles_num + dev_articles_num:]

        fname, ext = os.path.splitext(args.samples_file)
        train_file = fname + '-train' + ext
        dev_file = fname + '-dev' + ext
        test_file = fname + '-test' + ext

        train_samples_num = 0
        with open(train_file, 'w') as f:
            for article_sample in tqdm(train_articles_samples, total=len(train_articles_samples), desc='write train'):
                for sample in article_sample:
                    train_samples_num += 1
                    f.write(json.dumps(sample) + '\n')
        print(f'{train_samples_num} samples written to {train_file}.')
        dev_samples_num = 0
        with open(dev_file, 'w') as f:
            for article_sample in tqdm(dev_articles_samples, total=len(dev_articles_samples), desc='write dev'):
                for sample in article_sample:
                    dev_samples_num += 1
                    f.write(json.dumps(sample) + '\n')
        print(f'{dev_samples_num} samples written to {dev_file}.')
        test_samples_num = 0
        with open(test_file, 'w') as f:
            for article_sample in tqdm(test_articles_samples, total=len(test_articles_samples), desc='write test'):
                for sample in article_sample:
                    test_samples_num += 1
                    f.write(json.dumps(sample) + '\n')
        print(f'{test_samples_num} samples written to {test_file}.')
    else:
        samples_num = 0
        with open(args.samples_file, 'w') as fout:
            for article_sample in tqdm(samples, total=len(samples), desc='write total'):
                for sample in article_sample:
                    samples_num += 1
                    fout.write(json.dumps(sample) + '\n')
        print(f'{samples_num} samples written to {args.samples_file}.')

    print(colored(f'mismatched entities: {all_entities_mismatch}.', 'yellow'))
    print(colored(f'mismatched blame entities: {tie_entities_mismatch}', 'yellow'))

    print(f'{num_articles} articles generate {num_samples} samples.')
    print(f'avg entities per article: {mean(num_entities_dist):.2f} ± {stdev(num_entities_dist):.2f}')
    print(f'avg neg/pos ratio per article: {mean(samples_ratio_dist):.2f} ± {stdev(samples_ratio_dist):.2f}')
    print(f'pos: {num_pos_samples}, neg: {num_samples-num_pos_samples}, '
          f'neg / pos = {(num_samples-num_pos_samples)/num_pos_samples:.2f}, '
          f'overall pos percentage: {num_pos_samples / num_samples * 100:.2f}%')
    print(f'avg pair sentence distance: {mean(blame_tie_dist):.2f} ± {stdev(blame_tie_dist):.2f}')

    if args.save_stats:
        with open('num_entities_dist.txt', 'w') as f:
            f.write('\n'.join([str(e) for e in sorted(num_entities_dist, reverse=True)]))
        with open('samples_ratio_dist.txt', 'w') as f:
            f.write('\n'.join([str(e) for e in sorted(samples_ratio_dist, reverse=True)]))
        with open('blame_tie_dist.txt', 'w') as f:
            f.write('\n'.join([str(e) for e in sorted(blame_tie_dist, reverse=True)]))

    if args.plt_stats:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure()
        sns.distplot(num_entities_dist)
        plt.show()
        plt.figure()
        sns.distplot(samples_ratio_dist)
        plt.show()
        plt.figure()
        sns.distplot(blame_tie_dist)
        plt.show()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data for blame extractor')
    parser.register('type', 'bool', str2bool)
    # files
    parser.add_argument('--dataset-file', type=str, default='dataset.json')
    parser.add_argument('--samples-file', type=str, default='samples.json')
    # data
    parser.add_argument('--merge-entity', type=str, default='local',
                        help='build a merge dict locally(per article) or globally(dataset level)')
    parser.add_argument('--ignore-direction', type='bool', default=False)
    parser.add_argument('--split', type='bool', default=True)
    parser.add_argument('--all-entities', type='bool', default=False)

    # utility
    parser.add_argument('--data-force', type='bool', default=False)
    parser.add_argument('--save-stats', type='bool', default=False)
    parser.add_argument('--plt-stats', type='bool', default=False)
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
