#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 16:18:51
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-25 15:54:31

'''
Based on the output of script/blameextract/prepare_data.py
Generate samples in the format: ([ent-pos, ent-label] sents)
'''

import argparse
from collections import defaultdict
from collections import Counter
import os
import json
import pickle  # json cannot accpet tuple key
# from statistics import mean, stdev
# from termcolor import colored
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


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
    print(args)

    if args.merge_entity == 'global':
        entity2id = entity_merge_global(args)

    samples = []
    num_articles, num_samples = 0, 0
    labels_dist = Counter()
    with open(args.dataset_file) as f:
        lines = sum((1 for _ in f))
    with open(args.dataset_file) as f:
        for line in tqdm(f, total=lines, desc='generate entity samples'):
            d = json.loads(line)
            content = d['content']
            pairs = sorted({(tuple(p['source']), tuple(p['target'])) for p in d['pairs']})
            all_entities = sorted({tuple(e) for e in d['entities']})
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

            # # remove entities which cannot be found in article
            all_entities_ids = sorted({e for e in all_entities_ids if e in epos})
            pairs_entities_ids = sorted({(s, t) for s, t in pairs_entities_ids if s in epos and t in epos})

            if len(pairs_entities_ids) == 0:
                continue
            src_entities = {s for s, _ in pairs_entities_ids}
            tgt_entities = {t for _, t in pairs_entities_ids}

            sent_idxs = {si for entity in all_entities_ids for si, _ in epos[entity]}
            sents = [content[si] for si in sent_idxs]
            epos = {entity: [(sents.index(content[si]), wi) for si, wi in epos[entity]] for entity in all_entities_ids}
            entity_labels = []
            for e in all_entities_ids:
                if e in src_entities and e not in tgt_entities:
                    label = 'S'
                elif e not in src_entities and e in tgt_entities:
                    label = 'T'
                elif e in src_entities and e in tgt_entities:
                    label = 'ST'
                else:
                    label = 'N'
                entity_labels.append(label)
                labels_dist[label] += 1
                num_samples += 1

            sample = {'entities': all_entities_ids, 'labels': entity_labels, 'epos': epos, 'sents': sents}
            samples.append(sample)
            num_articles += 1

    print(f'{num_articles} articles generate {num_samples} entity samples.')
    print(f'{", ".join([label + ": " + str(count)  for label, count in labels_dist.items()])}')

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

        with open(train_file, 'w') as f:
            for article_sample in train_articles_samples:
                f.write(json.dumps(article_sample) + '\n')
        print(f'{len(train_articles_samples)} article samples written to {train_file}.')
        with open(dev_file, 'w') as f:
            for article_sample in dev_articles_samples:
                f.write(json.dumps(article_sample) + '\n')
        print(f'{len(dev_articles_samples)} article samples written to {dev_file}.')
        with open(test_file, 'w') as f:
            for article_sample in test_articles_samples:
                f.write(json.dumps(article_sample) + '\n')
        print(f'{len(test_articles_samples)} article samples written to {test_file}.')
    else:
        with open(args.samples_file, 'w') as fout:
            for article_sample in samples:
                fout.write(json.dumps(article_sample) + '\n')
        print(f'{len(samples)} article samples written to {args.samples_file}.')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data for blame entity classification')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--dataset-file', type=str, default='dataset.json')
    parser.add_argument('--samples-file', type=str, default='entity-class-samples.json')
    parser.add_argument('--merge-entity', type=str, default='local',
                        help='build a merge dict locally(per article) or globally(dataset level)')
    parser.add_argument('--split', type='bool', default=True)
    parser.add_argument('--data-force', type='bool', default=False)
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
