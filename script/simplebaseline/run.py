#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:14:09
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-22 16:14:43

"""Run the blame tie extractor baseline"""

import argparse
import json
import os
import sys
import logging
import subprocess
from collections import defaultdict

from termcolor import colored
import random
import numpy as np
from tqdm import tqdm
import torch

from blamepipeline import DATA_DIR as DATA_ROOT
from blamepipeline.simplebaseline import BaselineModel
from blamepipeline.simplebaseline import utils, config


logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults
DATA_DIR = os.path.join(DATA_ROOT, 'datasets')
LOG_DIR = os.path.join(DATA_ROOT, 'models/simplebaseline')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)
    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--random-seed', type=int, default=712,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--log-dir', type=str, default=LOG_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--train-file', type=str, default='samples.json',
                       help='train file')
    files.add_argument('--dev-file', type=str, default=None,
                       help='dev file')
    files.add_argument('--test-file', type=str, default=None,
                       help='test file')
    files.add_argument('--blame-lexicons', type=str, default='blame_lexicons.txt')
    # General
    general = parser.add_argument_group('General')
    general.add_argument('--metrics', type=str, choices=['precision', 'recall', 'F1', 'acc'],
                         help='metrics to display when training', nargs='+',
                         default=['precision', 'recall', 'F1', 'acc'])
    general.add_argument('--valid-metric', type=str, default='F1',
                         help='The evaluation metric used for model selection')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError(f'No such file: {args.train_file}')

    if args.dev_file:
        args.dev_file = os.path.join(args.data_dir, args.dev_file)
        if not os.path.isfile(args.dev_file):
            raise IOError(f'No such file: {args.dev_file}')

    if args.test_file:
        args.test_file = os.path.join(args.data_dir, args.test_file)
        if not os.path.isfile(args.test_file):
            raise IOError(f'No such file: {args.test_file}')

    if args.blame_lexicons:
        args.blame_lexicons = os.path.join(args.data_dir, args.blame_lexicons)
        if not os.path.isfile(args.blame_lexicons):
            raise IOError(f'No such file {args.blame_lexicons}')

    # Set log file names
        # Set model directory
    subprocess.call(['mkdir', '-p', args.log_dir])
    args.log_file = os.path.join(args.log_dir, 'baseline.txt')

    return args


def evaluate(pred, true, eps=1e-9):
    true_positive = (pred * true).sum().item()
    precision = true_positive / (pred.sum().item() + eps)
    recall = true_positive / (true.sum().item() + eps)
    F1 = 2 * (precision * recall) / (precision + recall + eps)
    acc = (pred == true).sum().item() / len(pred)
    return {'precision': precision, 'recall': recall, 'F1': F1, 'acc': acc}


def validate(args, data_loader, model, mode):
    """Run one full validation.
    """
    eval_time = utils.Timer()

    # Make predictions
    examples = 0
    preds = []
    trues = []

    for ex in tqdm(data_loader, total=len(data_loader), desc=f'validate {mode}'):
        batch_size = len(ex[-1])
        inputs = ex[:-1]
        pred = model.predict(inputs)
        true = ex[-1]

        preds += pred
        trues += true

        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    metrics = evaluate(np.array(preds), np.array(trues))

    logger.info(f'{mode} valid: ' +
                f'examples = {examples} | valid time = {eval_time.time():.2f} (s).')
    logger.info(' | '.join([f'{k}: {metrics[k]*100:.2f}%' for k in metrics]))

    return {args.valid_metric: metrics[args.valid_metric]}


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_data(args.train_file)
    logger.info(f'Num train examples = {len(train_exs)}')
    if args.dev_file:
        dev_exs = utils.load_data(args.dev_file)
        logger.info(f'Num dev examples = {len(dev_exs)}')
    else:
        dev_exs = []
        logger.info('No dev data. Randomly choose 10% of train data to validate.')
    if args.test_file:
        test_exs = utils.load_data(args.test_file)
        logger.info(f'Num test examples = {len(test_exs)}')
    else:
        test_exs = []
        logger.info('No test data. Use 10 fold cv to evaluate.')
    logger.info(f'Total {len(train_exs) + len(dev_exs) + len(test_exs)} examples.')

    logger.info(f'Loading blame lexicons from {args.blame_lexicons}...')
    with open(args.blame_lexicons) as f:
        lexicons = [w.strip().lower() for w in f.read().strip().split(' or ')]

    logging.info(f'{len(lexicons)} blame lexicons loaded.')

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))
    # --------------------------------------------------------------------------
    # DATA ITERATORS
    logger.info('-' * 100)
    logger.info('Make data loaders')
    if args.test_file:
        model = BaselineModel(config.get_model_args(args), lexicons)
        train_loader, dev_loader, test_loader = utils.split_loader(train_exs, test_exs, args, model,
                                                                   dev_exs=dev_exs)
        # Validate train
        validate(args, train_loader, model, mode='train')
        # Validate dev
        validate(args, dev_loader, model, mode='dev')
        # validate test
        result = validate(args, test_loader, model, mode='test')
        logger.info('-' * 100)
        logger.info(f'Test {args.valid_metric}: {result[args.valid_metric]*100:.2f}%')
    else:
        # 10-cross cv
        results = []
        samples_fold = [np.random.randint(10) for _ in range(len(train_exs))]
        fold_samples = defaultdict(list)
        for sample_idx, sample_fold in enumerate(samples_fold):
            fold_samples[sample_fold].append(sample_idx)
        model = BaselineModel(config.get_model_args(args), lexicons)
        for fold in range(10):
            fold_info = f'for fold {fold}' if fold is not None else ''
            logger.info(colored(f'Starting training {fold_info}...', 'blue'))
            train_loader, dev_loader = utils.split_loader_cv(
                train_exs, args, model, fold_samples[fold])
            result = validate(args, dev_loader, model, mode='dev')
            results.append(result[args.valid_metric])
            # logger.debug(colored('DEBUG: Run for 1 folds. Stop.', 'red'))
            # break
        result = np.mean(results).item()
        logger.info('-' * 100)
        logger.info(f'CV {args.valid_metric}: {result*100:.2f}%')


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Run Blame Extractor Baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set random state
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
