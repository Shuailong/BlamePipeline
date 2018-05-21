# coding: utf-8

'''
Load raw data, preprocess and save to file.

1. Group blame ties from the same article
2. Extract entities in an article
3. Tokenize entities and article contents

'''

import argparse
import os
import json
# import re
from collections import defaultdict
# from collections import Counter
from termcolor import colored
from tqdm import tqdm

from blamepipeline import DATA_DIR
from blamepipeline.preprocess.match_article_entry import match_data
from blamepipeline.preprocess.match_entity_article import filter_data
from blamepipeline.tokenizers import CoreNLPTokenizer

DATASET = os.path.join(DATA_DIR, 'datasets')


def contains(entity_toks, article_toks):
    entity_toks_ = entity_toks.copy()
    entity_toks_[-1] += '.'
    for s in article_toks:
        for i in range(len(s) - len(entity_toks) + 1):
            if s[i: i + len(entity_toks)] == entity_toks:
                return True
            elif s[i: i + len(entity_toks)] == entity_toks_:
                return True
    return False


def tokenize_entity(tokenizer, entity):
    '''
    Tokenize and clean the entity.
    entitt: str
    '''
    # tokenize
    entity = tuple(t for s in tokenizer.tokenize(entity).words() for t in s)
    # clean
    entity = entity[:-1] if entity[-1] == '.' else entity
    entity = entity[1:] if entity[0] == '--' else entity
    return entity


def main(args):
    print(args)
    if args.source == 'all':
        sources = ['USA', 'NYT', 'WSJ']
    else:
        sources = [args.source.upper()]
    # Acquire data
    data = []
    for source in sources:
        print('-' * 100)
        print(source)
        pairs = match_data(source)
        print('{} pairs loaded.'.format(len(pairs)))
        valid_pairs = filter_data(pairs, source=source, ignore_claim=args.ignore_claim)
        print('{} valid pairs.'.format(len(valid_pairs)))
        data += valid_pairs
    print(f'{len(data)} valid pairs in total.')
    tokenizer = CoreNLPTokenizer(annotators={'ner'})
    dataset_file = os.path.join(DATASET, 'blame', 'dataset.json')

    if not args.cluster_article:
        # don't need to cluster by article. use when training claim centence classification
        pbar = tqdm(data, desc='tokenize') if args.tqdm and args.tokenize else data
        with open(dataset_file, 'w') as f:
            for d in pbar:
                if args.tokenize:
                    if d['claim']:
                        d['claim'] = tokenizer.tokenize(d['claim']).words(uncased=args.uncased)
                    d['content'] = tokenizer.tokenize(d['content']).words(uncased=args.uncased)
                f.write(json.dumps(d) + '\n')
    else:
        # clustering by article
        articles_tie = defaultdict(list)
        articles_content = {}
        for d in data:
            tie = d['source'], d['target'], d['claim']
            key = d['title'], d['date']
            if tie not in articles_tie[key]:
                articles_tie[key].append({'source': d['source'], 'target': d['target'], 'claim': d['claim']})
            if key not in articles_content:
                articles_content[key] = d['content']
        print('-' * 100)
        print(f'{len(data)} valid pairs in {len(articles_tie)} articles.')

        # tokenize
        article_ents = {}
        if args.tokenize:
            # tokenize article
            for key in tqdm(articles_content, total=len(articles_content), desc='tokenize'):
                # for each article
                # tokenize content
                tokenized = tokenizer.tokenize(articles_content[key])
                articles_content[key] = tokenized.words(uncased=args.uncased)
                # automatically generated entities
                ner_entities = {name for name, tag in tokenized.entity_groups()
                                if tag in {'ORGANIZATION', 'PERSON'} and len(name) >= 2}
                # annotated entities
                anno_entities = {e for d in articles_tie[key] for e in (d['source'], d['target'])}
                # toknenize all_entities
                all_entities = {tokenize_entity(tokenizer, e) for e in ner_entities | anno_entities}

                for e in all_entities:
                    assert contains(list(e), articles_content[key]), colored(f'{e} not in {key}!', 'red')
                # combine entities
                article_ents[key] = list(all_entities)
                # tokenize blame tie entities
                articles_tie[key] = [{'source': tokenize_entity(tokenizer, d['source']),
                                      'target': tokenize_entity(tokenizer, d['target']),
                                      'claim': d['claim']}
                                     for d in articles_tie[key]]
        # write into file
        with open(dataset_file, 'w') as f:
            for key in articles_tie:
                title, date = key
                d = {'title': title,
                     'date': date,
                     'pairs': articles_tie[key],
                     'entities': article_ents[key],
                     'content': articles_content[key]}
                f.write(json.dumps(d) + '\n')
        print(f'Dataset saved to {dataset_file}')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make training data')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--source', type=str, default='all', choices=['nyt', 'wsj', 'usa', 'all'])
    parser.add_argument('--uncased', type='bool', default=False)
    parser.add_argument('--tokenize', type='bool', default=True)
    parser.add_argument('--ignore-claim', type='bool', default=True,
                        help='ignore existence of claim when filtering data entries.')
    parser.add_argument('--cluster-article', type='bool', default=True,
                        help='cluster blame ties in the same articles')
    parser.add_argument('--entity-anonymize', type='bool', default=False,
                        help='use entity id to represent entity name in claim and article content')
    # others
    parser.add_argument('--tqdm', type='bool', default=True)
    args = parser.parse_args()
    main(args)
