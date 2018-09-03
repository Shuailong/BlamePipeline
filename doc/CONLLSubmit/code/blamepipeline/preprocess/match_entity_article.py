# encoding: utf-8

'''
Match articles with annotated blame entities.
'''

import argparse
from collections import defaultdict

from .match_article_entry import match_data

sources = ['USA', 'NYT', 'WSJ']

cases = defaultdict(int)


def filter_data(pairs, source=None, ignore_claim=False):
    valid_pairs = []
    articles = set()
    global cases

    for entry, article in pairs:
        source = entry['source']
        target = entry['target']
        claim = entry['claim']
        content = article['content']

        if not source or not target:
            # empty entity
            cases['empty entity'] += 1
            continue
        if source.isdigit() or target.isdigit():
            # entity is number
            cases['digit entity'] += 1
            continue
        if len(source) < 2 or len(target) < 2:
            # entity length too short
            cases['entity too short'] += 1
            continue
        if source == target:
            # source and target is the same
            cases['same src and tgt'] += 1
            continue
        if source not in content:
            cases['src not in content'] += 1
            continue
        if target not in content:
            cases['tgt not in content'] += 1
            continue
        if not ignore_claim:
            if not claim:
                # no claim
                cases['no claim'] += 1
                continue
            if source not in claim:
                cases['src not in claim'] += 1
                continue
            if target not in claim:
                cases['tgt not in claim'] += 1
                continue
            if claim not in content:
                cases['claim not in content'] += 1
                continue
        d = {}
        d['title'] = entry['title']
        d['date'] = entry['date']
        d['source'] = source
        d['target'] = target
        d['claim'] = claim
        d['content'] = content
        valid_pairs.append(d)
        articles.add((entry['date'], entry['title']))
    print(f'{len(articles)} articles.')
    return valid_pairs


def main(args):
    print(args)
    data = []
    for source in sources:
        print(source)
        pairs = match_data(source)
        print('{} pairs loaded.'.format(len(pairs)))
        valid_pairs = filter_data(pairs, source=source, ignore_claim=args.ignore_claim)
        print('{} valid pairs.'.format(len(valid_pairs)))
        data += valid_pairs
        print('=-=-=-=-=-=')

    print('\n---\n')
    print('\n'.join([f'{k}: {cases[k]}' for k in cases]))


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract matched source and target in articles')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--ignore-claim', type='bool', default=False,
                        help='ignore existence of claim when filtering data entries.')
    args = parser.parse_args()
    main(args)
