#!/usr/bin/env python
# encoding: utf-8

"""
dataloader.py

Created by Shuailong on 2016-10-12.

Dataset for Blame Analysis project.

"""

import os
import csv
import re
import datetime
import string

from .. import DATA_DIR

BLAME_DATA = os.path.join(DATA_DIR, 'blamedata')

NYT_PATH = os.path.join(BLAME_DATA, 'NYT')
USA_PATH = os.path.join(BLAME_DATA, 'USA')
WSJ_PATH = os.path.join(BLAME_DATA, 'WSJ')


# Article structure for USA today, New York Times, and Wall Street Journal
USA_PATTERN = r'USA TODAY\s*(?P<date>\w+ \d+, \d+ \w+).*'\
              r'\s*\w+ EDITION\s*(?P<title>^[^\n]*$)'\
              r'\s*(?P<subtitle>(?!^BYLINE.*$)^[^\n]*$)'\
              r'\s*(BYLINE: (?P<author>[^\n]*))?'\
              r'\s*SECTION: ((?P<section>[\w]+); )?Pg\. \w+'\
              r'\s*LENGTH: (?P<length>\d+ words)'\
              r'\s*(?P<content>.*)LOAD-DATE'

NYT_PATTERN = r'The New York Times\s*'\
              r'(?P<date>\w+ \d+, \d+ \w+)\s*'\
              r'(^.*$\s*)?'\
              r'((\w+ Edition[^\n]*)|(The New York Times on the Web))\s*'\
              r'(?P<title>(?!((BYLINE)|(SECTION)).*$)^[^\n]*$)\s*'\
              r'(BYLINE: (?P<author>.*))?\s*'\
              r'SECTION: Section \w*; Column \w*;\s(?P<section>.*); Pg\. \w*\s*'\
              r'LENGTH: (?P<length>\d+ words)\s*'\
              r'(?:DATELINE:(?:[^\n]*$\s*))?(?P<content>.*)URL:'

WSJ_PATTERN = r'(\s*\bSE\b\s*)?'\
              r'(?P<title>(?!(HD))[^\n]*)\s*'\
              r'(\bHD\b\s*)?'\
              r'(?P<subtitle>(?!^(By.*)|(\w+ words)$)^[^\n]*)\s*'\
              r'(\bBY\b\s*)?'\
              r'(By (?P<author>[^\n]*$)\s*)?'\
              r'(\bWC\b\s*)?'\
              r'(?P<length>[\d,]+ words$)\s*'\
              r'(\bPD\b\s*)?'\
              r'(?P<date>\d+ \w+ \d+)\s*'\
              r'(\bSN\b\s*)?'\
              r'The Wall Street Journal\s*'\
              r'(\bSC\b\s*)?'\
              r'(?P<section>\w+)\s*'\
              r'(\bPG\b\s*)?'\
              r'(\w+\s*)?'\
              r'(\bLA\b\s*)?'\
              r'(English\s*)?'\
              r'(\bCY\b\s*)?'\
              r'\(Copyright \(c\) \d+, Dow Jones & Company, Inc\.\)\s*'\
              r'(\bLP\b\s*)?'\
              r'(?P<content>.*)License this article from Dow Jones Reprint Service\s*'\
              r'(?:.*)\s*'\
              r'Document (?P<id>\w+)'

# compile regex
PATTERNS = {'USA': USA_PATTERN, 'NYT': NYT_PATTERN, 'WSJ': WSJ_PATTERN}
REGEX = {source: re.compile(
    PATTERNS[source], re.DOTALL | re.MULTILINE) for source in PATTERNS}

# 4 October 2007
WSJ_DATE_FORMAT = '%d %B %Y'
# June 29, 2010 Tuesday
USA_NYT_DATE_FORMAT = '%B %d, %Y %A'


class DBReader():
    '''
    A class to read from csv file.
    Input: filename
    Output: List[{Entry}]
    Entry: {title, date, section, source, s_title, s_category, target, t_title,
            t_category, claim}
    '''

    def __init__(self, filename, source):
        # print('Reading from {}...'.format(filename))
        self.source = source
        self.entries = []
        with open(filename) as csv_file:
            fieldnames = ['valid', 'title', 'date', 'source', 'target', 'claim']
            reader = csv.DictReader(csv_file, fieldnames=fieldnames, dialect="excel")
            next(reader)
            for row in reader:
                if row['valid'] == '0':
                    continue
                r = {}
                r['title'] = row['title'].strip().lstrip(', ')
                r['date'] = row['date'].strip()
                r['source'] = row['source'].strip()
                r['target'] = row['target'].strip()
                r['claim'] = row['claim'].strip().lstrip(string.punctuation).replace("''", '"')
                if r not in self.entries:
                    self.entries.append(r)

    def get_entries(self):
        '''
        Return all the entries in the dataset.
        '''
        return self.entries


class Articles():
    '''
    A list of news articles.
    Input: filename
    Output: List[{Article}]
    Article: {date, title, subtitle, author, section, length, content}
    '''

    def __init__(self, filename, source):
        # read articles and match
        self.source = source
        self.articles = []
        with open(filename) as article_file:
            raw_articles = article_file.read().strip().split('\f')
            for raw_article in raw_articles:
                raw_article = raw_article.strip()
                article = self.match(raw_article)
                if article and (article['title'] or article['subtitle']):
                    if article not in self.articles:
                        self.articles.append(article)

    def match(self, raw_article):
        '''
        Using regex to extract information
        raw_article: str
        return: {date, title, subtitle, author, section, length, content}
        '''

        regex = REGEX[self.source]
        search_res = regex.search(raw_article)

        article = {}
        if search_res:
            date = search_res.group('date')
            if date:
                date = self.reformat_date(date)

            title = search_res.group('title')
            if title:
                title = title.strip()
                if title.endswith(';'):
                    title = title[:-1]
            if self.source == 'NYT':
                subtitle = ''  # no such data in NYT database
            else:
                subtitle = search_res.group('subtitle')
                if subtitle:
                    subtitle = subtitle.strip()
            author = search_res.group('author')
            if author:
                author = author.strip()
            section = search_res.group('section')
            if section:
                section = section.strip()
            length = search_res.group('length')
            content = search_res.group('content')
            if content:
                content = content.strip().replace("''", '"')

            article = {'date': date, 'title': title, 'subtitle': subtitle,
                       'author': author, 'section': section, 'length': length,
                       'content': content}
        else:
            article = None

        return article

    def reformat_date(self, date):
        '''
        format date string to 'YYYYmmdd'
        '''
        if self.source == 'USA' or self.source == 'NYT':
            old_format = USA_NYT_DATE_FORMAT
        elif self.source == 'WSJ':
            old_format = WSJ_DATE_FORMAT
        reformat = '%Y%m%d'
        date = datetime.datetime.strptime(date, old_format).strftime(reformat)
        return date

    def get_articles(self):
        '''
        Return all the articles in the dataset.
        '''
        return self.articles


class Dataset():
    '''A class interface to access datasets'''

    def __init__(self, source):
        dataset_dir = {'NYT': NYT_PATH, 'USA': USA_PATH, 'WSJ': WSJ_PATH}
        if source not in dataset_dir:
            raise ValueError(
                '{} is not supported yet. Consider {} instead'.format(
                    source,
                    dataset_dir.keys()))
        self.source = source
        self.dirname = dataset_dir[source]
        self.files = os.listdir(self.dirname)
        self.files.sort()

    def get_entries(self):
        '''
        Return entries of the dataset.
        '''

        filename = self.source + '_blame_relation_database.csv'
        entries = DBReader(os.path.join(self.dirname, filename), self.source).get_entries()

        return entries

    def get_articles(self):
        '''
        Return articles of the dataset.
        '''
        article_data = []
        for filename in self.files:
            if filename.endswith('.txt'):
                article_data += Articles(
                    os.path.join(self.dirname, filename),
                    self.source).get_articles()

        return article_data


def main():
    '''
    Entry of the program.
    '''
    print('Dataset statistics:')
    datasetnames = ['USA', 'NYT', 'WSJ']

    for datasetname in datasetnames:
        print('\n{}:'.format(datasetname))
        dataset = Dataset(datasetname)
        entries = dataset.get_entries()
        articles = dataset.get_articles()
        print('{} articles. {} entries.'.format(len(articles), len(entries)))


if __name__ == '__main__':
    main()
