# coding: utf-8

'''
Load raw data, preprocess and save to file.
'''

import argparse
import os
import json
import re
from collections import defaultdict
from collections import Counter

from tqdm import tqdm

from blamepipeline import DATA_DIR
from blamepipeline.preprocess.match_article_entry import match_data
from blamepipeline.preprocess.match_entity_article import filter_data
from blamepipeline.tokenizers import CoreNLPTokenizer

DATASET = os.path.join(DATA_DIR, 'datasets')


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
    tokenizer = CoreNLPTokenizer()
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
                articles_tie[key].append(tie)
            if key not in articles_content:
                articles_content[key] = d['content']
        print('-' * 100)
        print(f'{len(data)} valid pairs in {len(articles_tie)} articles.')

        # in one article, if one entity is part of another entity, use another entity as this entity
        for key in articles_tie:
            ents = {e for (s, t, _) in articles_tie[key] for e in (s, t)}
            for i, (s, t, c) in enumerate(articles_tie[key]):
                for e in ents:
                    if s in e and s != e:
                        articles_tie[key][i] = e, t, c
                        break
                    if t in e and t != e:
                        articles_tie[key][i] = s, e, c
                        break
        # use entity id to represent an annotated entity
        if args.entity_anonymize:
            ents = [e for key in articles_tie for (s, t, _) in articles_tie[key] for e in (s, t)]
            entity_count = Counter(ents)
            entity_dict = {}
            entity_file = os.path.join(DATASET, 'blame', 'entity.json')
            with open(entity_file, 'w') as f:
                for idx, (ent, freq) in enumerate(entity_count.most_common()):
                    ent_id = f'ENTITYID_{str(idx)}'
                    d = {'idx': idx,
                         'entity': ent,
                         'freq': freq,
                         'id': ent_id,
                         }
                    entity_dict[ent] = d
                    f.write(json.dumps(d) + '\n')
            for key in articles_tie:
                for i, (s, t, c) in enumerate(articles_tie[key]):
                    articles_tie[key][i] = entity_dict[s]['id'], entity_dict[t]['id'], c
                    articles_content[key] = articles_content[key].replace(s, entity_dict[s]['id'])
                    articles_content[key] = articles_content[key].replace(t, entity_dict[t]['id'])

        # tokenize
        if args.tokenize:
            # tokenize article
            for key in tqdm(articles_content, total=len(articles_content), desc='tokenize'):
                articles_content[key] = tokenizer.tokenize(articles_content[key]).words(uncased=args.uncased)
                # fix entity tokenization bugs
                for si, s in enumerate(articles_content[key]):
                    ss = []
                    for wi, w in enumerate(s):
                        res = re.match(r'(.*)(entityid_\d+)(.*)', w)
                        if res:
                            leftcontext, e, rightcontext = res.group(1), res.group(2), res.group(3)
                            if leftcontext:
                                resl = re.match(r'(.*)(entityid_\d+)(.*)', leftcontext)
                                if resl:
                                    leftcontextl, el, rightcontextl = resl.group(1), resl.group(2), resl.group(3)
                                    if leftcontextl:
                                        ss.append(leftcontextl)
                                    ss.append(el)
                                    if rightcontextl:
                                        ss.append(rightcontextl)
                                ss.append(leftcontext)
                            ss.append(e)
                            if rightcontext:
                                ss.append(rightcontext)
                        else:
                            ss.append(w)
                    articles_content[key][si] = ss
            # tokenize claim
            for key in articles_tie:
                for i, (s, t, c) in enumerate(articles_tie[key]):
                    if args.uncased:
                        s, t = s.lower(), t.lower()
                    if c:
                        c = tokenizer.tokenize(c).words(uncased=args.uncased)
                    articles_tie[key][i] = {'source': s, 'target': t, 'claim': c}
        # write into file
        with open(dataset_file, 'w') as f:
            for key in articles_tie:
                title, date = key
                d = {'title': title, 'date': date, 'pairs': articles_tie[key], 'content': articles_content[key]}
                f.write(json.dumps(d) + '\n')
        print(f'Dataset saved to {dataset_file}')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make training data')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--source', type=str, default='all', choices=['nyt', 'wsj', 'usa', 'all'])
    parser.add_argument('--uncased', type='bool', default=True)
    parser.add_argument('--tokenize', type='bool', default=True)
    parser.add_argument('--ignore-claim', type='bool', default=True,
                        help='ignore existence of claim when filtering data entries.')
    parser.add_argument('--cluster-article', type='bool', default=True,
                        help='cluster blame ties in the same articles')
    parser.add_argument('--entity-anonymize', type='bool', default=True,
                        help='use entity id to represent entity name in claim and article content')
    # others
    parser.add_argument('--tqdm', type='bool', default=True)
    args = parser.parse_args()
    main(args)
