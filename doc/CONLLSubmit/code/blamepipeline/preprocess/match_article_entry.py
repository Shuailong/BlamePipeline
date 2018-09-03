# encoding: utf-8

'''
Match articles with annotated data
'''

from collections import defaultdict
import argparse

from .dataloader import Dataset

case1, case2 = 0, 0


def match_data(source):
    dataset = Dataset(source)
    articles = dataset.get_articles()
    entries = dataset.get_entries()

    date_articles = defaultdict(list)
    for article in articles:
        date_articles[article['date']].append(article)
    print('{} dates of {} articles loaded.'.format(len(date_articles), len(articles)))
    print('{} entries loaded.'.format(len(entries)))

    title_match = 0
    subtitle_match = 0

    pairs = []

    def matches(entry_title, article_title):
        if not entry_title or len(entry_title) < 10:
            return False
        elif entry_title and article_title and entry_title == article_title:
            return True
        elif entry_title and entry_title in article_title:
            return True
        return False

    for entry in entries:
        for article in date_articles[entry['date']]:
            if matches(entry['title'], article['title']):
                title_match += 1
                pairs.append((entry, article))
                break
            elif matches(entry['title'], article['subtitle']):
                subtitle_match += 1
                pairs.append((entry, article))
                break

    print('title match:', title_match)
    print('subtitle match:', subtitle_match)

    return pairs


def main(args):
    if args.source == 'all':
        sources = ['USA', 'NYT', 'WSJ']
    else:
        sources = [args.source.upper()]
    for source in sources:
        print(source)
        pairs = match_data(source)
        print('matched pairs:', len(pairs))
        print('---')
    global case1, case2
    print(f'{case1}, {case2}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='match articles and entries')
    parser.add_argument('--source', type=str, choices=['all', 'wsj', 'nyt', 'usa'], default='all')
    args = parser.parse_args()
    main(args)
