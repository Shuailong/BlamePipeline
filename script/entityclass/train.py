#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:14:09
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-30 20:32:08

"""Train the blame tie extractor"""

import argparse
import json
import os
import sys
import subprocess
import logging
from collections import defaultdict

from termcolor import colored
import random
import numpy as np
import torch
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger

from blamepipeline import DATA_DIR as DATA_ROOT
from blamepipeline.entityclass import EntityClassifier
from blamepipeline.entityclass import utils, config


logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults
DATA_DIR = os.path.join(DATA_ROOT, 'datasets')
MODEL_DIR = os.path.join(DATA_ROOT, 'models/entityclass')
EMBED_DIR = os.path.join(DATA_ROOT, 'embeddings')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=0,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=0,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=712,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=30,
                         help='Train data iterations')
    runtime.add_argument('--early-stopping', type=int, default=10,
                         help='Early stopping patience')
    runtime.add_argument('--batch-size', type=int, default=5,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=5,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str, default='entity-class-samples-train-binary.json',
                       help='train file')
    files.add_argument('--dev-file', type=str, default='entity-class-samples-dev-binary.json',
                       help='dev file')
    files.add_argument('--test-file', type=str, default='entity-class-samples-test-binary.json',
                       help='test file')
    files.add_argument('--stats-file', type='bool', default=False,
                       help='store training stats in to file for display in codalab')
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--pretrain-file', type=str, choices=['w2v', 'glove', 'elmo'],
                       default=None, help='pretrained embeddings file/elmo')
    files.add_argument('--valid-size', type=float, default=0.1,
                       help='validation set ratio')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--metrics', type=str, choices=['precision', 'recall', 'F1', 'acc'],
                         help='metrics to display when training', nargs='+',
                         default=['F1', 'precision', 'recall', 'acc'])
    general.add_argument('--valid-metric', type=str, default='recall',
                         help='The evaluation metric used for model selection')
    general.add_argument('--uncased', type='bool', default=True,
                         help='uncase data')
    general.add_argument('--vocab-cutoff', type=int, default=1,
                         help='word frequency larger than this will be in dictionary')
    general.add_argument('--visdom', type='bool', default=False,
                         help='Use visdom to visualize loss etc.')
    general.add_argument('--visdom-port', type=int, default=9707,
                         help='Visdom port number')

    # debug
    debug = parser.add_argument_group('Debug')
    debug.add_argument('--debug', type='bool', default=False,
                       help='Debug mode: show dev and test data vocabulary coverage and use small portion of data.')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)

    if args.dev_file:
        args.dev_file = os.path.join(args.data_dir, args.dev_file)
        if not os.path.isfile(args.dev_file):
            raise IOError('No such file: %s' % args.dev_file)

    if args.test_file:
        args.test_file = os.path.join(args.data_dir, args.test_file)
        if not os.path.isfile(args.test_file):
            raise IOError('No such file: %s' % args.test_file)

    if args.pretrain_file:
        if args.pretrain_file in ['w2v', 'glove']:
            if args.pretrain_file == 'w2v':
                args.pretrain_file = f'w2v.googlenews.{args.embedding_dim}d.txt'
            else:
                args.pretrain_file = f'glove.6B.{args.embedding_dim}d.txt'
            args.pretrain_file = os.path.join(args.embed_dir, args.pretrain_file)
            if not os.path.isfile(args.pretrain_file):
                raise IOError(f'No such file: {args.pretrain_file}')
        elif args.pretrain_file == 'elmo':
            # args.pretrain_file == 'elmo':
            args.elmo_weights_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            args.elmo_options_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
            args.elmo_weights_file = os.path.join(args.embed_dir, args.elmo_weights_file)
            args.elmo_options_file = os.path.join(args.embed_dir, args.elmo_options_file)
            if not os.path.isfile(args.elmo_weights_file):
                raise IOError(f'No such file: {args.elmo_weights_file}')
            if not os.path.isfile(args.elmo_options_file):
                raise IOError(f'No such file: {args.elmo_options_file}')
    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    if args.stats_file:
        args.stats_file = os.path.join(args.model_dir, 'stats')

    # Embeddings options
    if args.pretrain_file and ('glove' in args.pretrain_file or 'w2v' in args.pretrain_file):
        with open(args.pretrain_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either pretrain_file or embedding_dim '
                           'needs to be specified.')

    # Make sure fix_embeddings and pretrain_file are consistent
    if args.fix_embeddings:
        if not args.pretrain_file:
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs, test_exs):
    """New model, new data, new dictionary.
    """

    # Build a dictionary from the data
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict(args, train_exs + dev_exs + test_exs, cutoff=args.vocab_cutoff)
    logger.info(f'Num words = {len(word_dict)}')
    label_dict = utils.build_label_dict(args, train_exs + dev_exs + test_exs)
    logger.info(f'Num labels = {len(label_dict)}')

    # Initialize model
    model = EntityClassifier(config.get_model_args(args), word_dict, label_dict)

    # Load pretrained embeddings for words in dictionary
    if args.pretrain_file and ('glove' in args.pretrain_file or 'w2v' in args.pretrain_file):
        model.load_embeddings(word_dict.tokens(), args.pretrain_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------

def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()
    train_loss_overall = utils.AverageMeter()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        loss, batch_size = model.update(ex)
        train_loss.update(loss, batch_size)
        train_loss_overall.update(loss, batch_size)

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))
    return train_loss_overall.avg

# ------------------------------------------------------------------------------
# Validation loops. Includes functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------


def evaluate(pred, true, eps=1e-9):
    assert len(pred) == len(true)
    true_positive = (pred * true).sum().item()
    precision = true_positive / (pred.sum().item() + eps)
    recall = true_positive / (true.sum().item() + eps)
    F1 = 2 * (precision * recall) / (precision + recall + eps)
    acc = (pred == true).sum().item() / len(pred)
    return {'precision': precision, 'recall': recall, 'F1': F1, 'acc': acc}


def validate(args, data_loader, model, global_stats, mode, confusion_meter=None):
    """Run one full validation.
    """
    eval_time = utils.Timer()

    # Make predictions
    examples = 0
    preds = []
    trues = []

    for ex in data_loader:
        batch_size = ex[-1].size(0)
        inputs = ex[:-1]
        pred = model.predict(inputs)
        true = ex[-1]

        preds.append(pred)
        trues.append(true)

        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    preds = preds.cpu().data.numpy()
    trues = trues.cpu().numpy()

    metrics = {m: v for m, v in evaluate(preds, trues).items() if m in args.metrics}

    cm = None
    if confusion_meter:
        confusion_meter.add(preds, trues)
        cm = confusion_meter.value()

    logger.info(f'{mode} valid: Epoch = {global_stats["epoch"]} (best:{global_stats["best_epoch"]}) | ' +
                f'examples = {examples} | valid time = {eval_time.time():.2f} (s).')
    test_result = ' | '.join([f'{k}: {metrics[k]*100:.2f}%' for k in metrics])
    if mode == 'test':
        test_result = colored(test_result, 'green')
    logger.info(test_result)

    return metrics, cm


def train_valid_loop(train_loader, dev_loader, test_loader, args, model, fold=None):
    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0, 'best_epoch': 0, 'fold': fold}
    start_epoch = 0

    if args.visdom:
        # add visdom logger code
        port = args.visdom_port
        train_loss_logger = VisdomPlotLogger(
            'line', port=port, opts={'title': f'{args.model_name} Train Loss'})
        train_metric_logger = VisdomPlotLogger(
            'line', port=port, opts={'title': f'{args.model_name} Train Class Accuracy'})
        idx2label = {i: label for label, i in model.label_dict.items()}
        label_names = [idx2label[i] for i in range(model.args.label_size)]
        train_confusion_logger = VisdomLogger('heatmap',
                                              port=port,
                                              opts={'title': f'{args.model_name} Train Confusion Matrix',
                                                    'columnnames': label_names,
                                                    'rownames': label_names})
        valid_metric_logger = VisdomPlotLogger(
            'line', port=port, opts={'title': f'{args.model_name} Valid Class Accuracy'})
        valid_confusion_logger = VisdomLogger('heatmap',
                                              port=port,
                                              opts={'title': f'{args.model_name} Valid Confusion Matrix',
                                                    'columnnames': label_names,
                                                    'rownames': label_names})
        train_confusion_meter = tnt.meter.ConfusionMeter(model.args.label_size, normalized=True)
        valid_confusion_meter = tnt.meter.ConfusionMeter(model.args.label_size, normalized=True)
    else:
        train_confusion_meter = None
        valid_confusion_meter = None

    try:
        for epoch in range(start_epoch, args.num_epochs):
            stats['epoch'] = epoch

            # Train
            loss = train(args, train_loader, model, stats)
            stats['train_loss'] = loss

            # Validate train
            train_res, train_cfm = validate(args, train_loader, model, stats,
                                            mode='train', confusion_meter=train_confusion_meter)
            for m in train_res:
                stats['train_' + m] = train_res[m]

            # Validate dev
            val_res, valid_cfm = validate(args, dev_loader, model, stats, mode='dev',
                                          confusion_meter=valid_confusion_meter)
            for m in train_res:
                stats['dev_' + m] = val_res[m]

            if args.visdom:
                train_loss_logger.log(epoch, loss)
                train_metric_logger.log(epoch, train_res[args.valid_metric])
                train_confusion_logger.log(train_cfm)

                valid_metric_logger.log(epoch, val_res[args.valid_metric])
                valid_confusion_logger.log(valid_cfm)

                train_confusion_meter.reset()
                valid_confusion_meter.reset()

            # Save best valid
            if val_res[args.valid_metric] > stats['best_valid']:
                logger.info(
                    colored(f'Best valid: {args.valid_metric} = {val_res[args.valid_metric]*100:.2f}% ', 'yellow') +
                    colored(f'(epoch {stats["epoch"]}, {model.updates} updates)', 'yellow'))
                fold_info = f'.fold_{fold}' if fold is not None else ''
                model.save(args.model_file + fold_info)
                stats['best_valid'] = val_res[args.valid_metric]
                stats['best_epoch'] = epoch
            logger.info('-' * 100)

            if args.stats_file:
                with open(args.stats_file, 'w') as f:
                    out_stats = stats.copy()
                    out_stats['timer'] = out_stats['timer'].time()
                    if fold is None:
                        del out_stats['fold']
                    f.write(json.dumps(out_stats) + '\n')

            if epoch - stats['best_epoch'] >= args.early_stopping:
                logger.info(colored(f'No improvement for {args.early_stopping} epochs, stop training.', 'red'))
                break
    except KeyboardInterrupt:
        logger.info(colored(f'User ended training. stop.', 'red'))

    logger.info('Load best model...')
    model = EntityClassifier.load(args.model_file + fold_info, args)
    # device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    # model.to(device)
    model.cuda()
    stats['epoch'] = stats['best_epoch']
    if fold is not None:
        mode = f'fold {fold} test'
    else:
        mode = 'test'
    test_result, _ = validate(args, test_loader, model, stats, mode=mode)
    return test_result


def initialize_model(train_exs, dev_exs, test_exs):
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    logger.info('Training model from scratch...')
    model = init_from_scratch(args, train_exs, dev_exs, test_exs)
    # Set up optimizer
    model.init_optimizer()

    # Use the GPU?
    # device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    # model.to(device)
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()
    return model

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
        if args.debug:
            train_exs = train_exs[:10]
            dev_exs = dev_exs[:3]
            test_exs = test_exs[:3]
        model = initialize_model(train_exs, dev_exs, test_exs)
        train_loader, dev_loader, test_loader = utils.split_loader(train_exs, test_exs, args, model,
                                                                   dev_exs=dev_exs)
        result = train_valid_loop(train_loader, dev_loader, test_loader, args, model)[args.valid_metric]
        logger.info('-' * 100)
        logger.info(f'Test {args.valid_metric}: {result*100:.2f}%')
    else:
        # 10-cross cv
        results = []
        samples_fold = [np.random.randint(10) for _ in range(len(train_exs))]
        fold_samples = defaultdict(list)
        for sample_idx, sample_fold in enumerate(samples_fold):
            fold_samples[sample_fold].append(sample_idx)
        for fold in range(10):
            fold_info = f'for fold {fold}' if fold is not None else ''
            logger.info(colored(f'Starting training {fold_info}...', 'blue'))
            model = initialize_model(train_exs, dev_exs, test_exs)
            train_loader, dev_loader, test_loader = utils.split_loader_cv(
                train_exs, args, model, fold_samples[fold])
            result = train_valid_loop(train_loader, dev_loader, test_loader, args, model, fold=fold)
            results.append(result[args.valid_metric])
            if args.debug:
                # DEBUG
                logger.debug(colored('DEBUG: Run for 1 folds. Stop.', 'red'))
                break
        result = np.mean(results).item()
        std = np.std(results).item()
        logger.info('-' * 100)
        logger.info(f'10 fold cross validation test {args.valid_metric}s: {results}')
        logger.info('Averaged test ' + colored(f'{args.valid_metric}: {result*100:.2f}±{std*100:.2f}%', 'green'))


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Train EntityClassifier BiLSTM Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set random state
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # this will **slower** the speed a lot, but enforce deterministic result for CNN model
    # torch.backends.cudnn.enabled = False

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
