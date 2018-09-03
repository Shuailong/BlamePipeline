# coding: utf-8

'''
Reformat dataset.json.
Randomly select 100 articles for interagreement annotation test.

'''


import os
import json
import random
import argparse

from blamepipeline import DATA_DIR


def main(args):
    print(args)
    random.seed(1997)
    ifile = os.path.join(DATA_DIR, 'datasets', args.input)
    ofile = os.path.join(DATA_DIR, 'datasets', args.output)
    ans_file = os.path.join(DATA_DIR, 'datasets', args.answer)

    with open(ifile) as rfile, open(ofile, 'w') as wfile, open(ans_file, 'w') as ans_f:
        articles = [json.loads(line) for line in rfile.readlines()]
        count = 0
        for doc_no, article in enumerate(articles):
            title = article['title']
            date = article['date']
            date = date[0:4] + '-' + date[4:6] + '-' + date[6:]
            pairs = article['pairs']
            entities = article['entities']
            content = article['content']
            entities = set()
            for pair in pairs:
                source = ' '.join(pair['source'])
                target = ' '.join(pair['target'])
                entities.add(source)
                entities.add(target)
            entities = list(entities)
            #     claim = pair['claim']
            #     wfile.write(f'\tSource: {source}\n')
            #     wfile.write(f'\tTarget: {target}\n')
            #     wfile.write(f'\tClaim: {claim}\n')
            #     wfile.write(f'\t---\n')

            filtered_sents = []
            for s in content:
                remains = False
                for e in entities:
                    if e in ' '.join(s):
                        remains = True
                        break
                if remains:
                    filtered_sents.append(' '.join(s))
            for e in entities:
                found = False
                for s in filtered_sents:
                    if e in s:
                        found = True
                        break
                if not found:
                    assert f'Entity {e} not found!'
            if len(filtered_sents) > args.sentence_limit:
                continue

            wfile.write(f'Document No. {count+1}\n')
            wfile.write(f'Title: {title}\n')
            wfile.write(f'Date: {date}\n')
            wfile.write('Entities: ')
            wfile.write(
                ', '.join([f'[{i+1}] {e}' for i, e in enumerate(entities)]) + '\n')
            wfile.write('Article:\n')
            filtered_sents = '\n'.join(
                [f'({i+1}) {s}' for i, s in enumerate(filtered_sents)])
            wfile.write(filtered_sents)
            wfile.write('\n\n===\n\n')

            # write answer
            ans_f.write(f'{count+1}\n')
            pair_strs = []
            for pair in pairs:
                source = ' '.join(pair['source'])
                target = ' '.join(pair['target'])
                s_i = entities.index(source)
                t_i = entities.index(target)
                pair_strs.append(f'{s_i+1}-{t_i+1}')
            ans_f.write(','.join(pair_strs) + '\n')

            count += 1

            if count >= args.samples:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Randomly select 100 articles for interagreement test.')
    parser.add_argument('--input', default='dataset.json')
    parser.add_argument('--output', default='dataset_agreement.txt')
    parser.add_argument('--answer', default='answer.txt')
    parser.add_argument('--samples', type=int, default=100,
                        help='number of sample articles')
    parser.add_argument('--sentence_limit', type=int, default=10,
                        help='maximum effective sentences in an article')
    args = parser.parse_args()
    main(args)
