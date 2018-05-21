#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 16:18:51
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-21 22:03:54

'''
Within an article, merge the same entities and construct a map from entity name to entity id.
In the blame tie and article content, replace entity names by its id.
'''

import argparse
from collections import defaultdict
import os
import json
import pickle  # json cannot accpet tuple key
# from termcolor import colored
from itertools import permutations
from tqdm import tqdm


def entity_merge_global(args):
    '''
    Merge entities globally.
    entity2alias: {longest entity name(tuple) ->all alias sort by word length in decrease order (tuples)}
    entity2id: {entity name (tuple) -> longest entity name (str) }
    '''
    entity_json_path = args.entity_dir + 'entity.json'
    entity_txt_path = args.entity_dir + 'entity.txt'
    entity2id_path = args.entity_dir + 'entity2id.pkl'

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
    articles, samples, pos = 0, 0, 0
    fname, ext = os.path.splitext(args.samples_file)
    if args.ignore_direction:
        args.samples_file = fname + '-undirected' + ext
    else:
        args.samples_file = fname + '-directed' + ext
    print(args)

    if args.merge_entity == 'global':
        entity2id = entity_merge_global(args)

    # debug
    all_entities_mismatch, tie_entities_mismatch, total_entities = 0, 0, 0
    with open(args.dataset_file) as f,\
            open(args.samples_file, 'w') as fout:
        for line in tqdm(f, total=999, desc='generate samples'):
            articles += 1
            d = json.loads(line)
            content = d['content']
            pairs = {(tuple(p['source']), tuple(p['target'])) for p in d['pairs']}
            all_entities = {tuple(e) for e in d['entities']}
            if args.merge_entity == 'local':
                entity2id = entity_merge_local(all_entities)
            pairs_entities_ids = {(entity2id[s], entity2id[t]) for s, t in pairs}
            all_entities_ids = {entity2id[e] for e in all_entities}

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
            # DEBUG
            for i, e in enumerate(all_entities_ids):
                if e not in epos:
                    all_entities_mismatch += 1
            for p in pairs_entities_ids:
                for e in p:
                    if e not in epos:
                        tie_entities_mismatch += 1
            total_entities += len(all_entities)

            # continue
            # # remove entities which cannot be found in article
            all_entities_ids = {e for e in all_entities_ids if e in epos}
            pairs_entities_ids = {(s, t) for s, t in pairs_entities_ids if s in epos and t in epos}

            # make negative samples and select sentences
            for src, tgt in permutations(all_entities_ids, 2):
                if not args.ignore_direction:
                    label = 1 if (src, tgt) in pairs_entities_ids else 0
                else:
                    label = 1 if (src, tgt) in pairs_entities_ids or (tgt, src) in pairs_entities_ids else 0
                if label == 1:
                    pos += 1
                sent_idxs = {si for si, _ in epos[src] + epos[tgt]}
                sents = [content[si] for si in sent_idxs]
                s_pos = [(sents.index(content[si]), wi) for si, wi in epos[src]]
                t_pos = [(sents.index(content[si]), wi) for si, wi in epos[tgt]]
                dpos = {'src_pos': s_pos, 'tgt_pos': t_pos, 'sents': sents, 'label': label}
                fout.write(json.dumps(dpos) + '\n')
                samples += 1

    print(f'mismatched entities: {all_entities_mismatch}.')
    print(f'mismatched blame entities: {tie_entities_mismatch}')
    print(f'total entities: {total_entities}')
    print(f'{articles} articles generate {samples} samples.')
    print(f'neg / pos = {(samples - pos) / pos:.2f}, pos percentage: {pos / samples * 100:.2f}%')

    if args.split:
        pass


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data for blame extractor')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--dataset-file', type=str, default='data/datasets/blame/dataset.json')
    parser.add_argument('--entity-dir', type=str, default='data/datasets/blame/')
    parser.add_argument('--samples-file', type=str, default='data/datasets/blame/samples.json')
    parser.add_argument('--merge-entity', type=str, default='local',
                        help='build a merge dict locally(per article) or globally(dataset level)')
    parser.add_argument('--ignore-direction', type='bool', default=False)
    parser.add_argument('--split', type='bool', default=True)
    parser.add_argument('--data-force', type='bool', default=False)
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
