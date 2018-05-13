#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-10 20:16:42

'''
BlameExtractor Class Wrapper
'''


import logging
from .extractor import LexiconClassifier


logger = logging.getLogger(__name__)


class BlameExtractor(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, lexicons):
        # Book-keeping.
        self.args = args

        # Building network.
        if args.model_type == 'lexicon':
            self.classifier = LexiconClassifier(args, lexicons, mode=args.mode)
        else:
            raise RuntimeError(f'Unsupported model: {args.model_type}')

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, inputs):
        pred = self.classifier.predict(inputs)
        return pred
