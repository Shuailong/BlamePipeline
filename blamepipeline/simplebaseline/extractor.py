#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:42:04
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-09-04 01:06:50
"""Implementation of the Blame Extractor Baseline Class."""

# ------------------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------------------
import random
import logging

logger = logging.getLogger(__name__)


class LexiconClassifier(object):
    def __init__(self, args, lexicons, mode=None):
        super(LexiconClassifier, self).__init__()
        if mode is None:
            mode = 'default'
        logger.info(f'Initialize baseline model in {mode}')
        self.args = args
        self.lexicons = lexicons
        self.mode = mode

    def predict(self, ex):
        batch_spos, batch_sapos, batch_tpos, batch_tapos, batch_sents = ex
        predicts = []
        for spos, sapos, tpos, tapos, sents in zip(batch_spos, batch_sapos, batch_tpos, batch_tapos, batch_sents):
            sents = [' '.join(s) for s in sents]
            if self.mode == 'mode1':
                label = self._mode1(sapos, tapos)
            elif self.mode == 'mode2':
                label = self._mode2(sapos, tapos)
            elif self.mode == 'mode3':
                label = self._mode3(spos, tpos, sents)
            elif self.mode == 'mode1+mode3':
                label = 1 if self._mode1(sapos, tapos) and self._mode3(spos, tpos, sents) else 0
            elif self.mode == 'mode2+mode3':
                label = 1 if self._mode2(sapos, tapos) and self._mode3(spos, tpos, sents) else 0
            else:
                # random
                label = random.randint(0, 1)
            predicts.append(label)
        return predicts

    def _mode1(self, spos, tpos):
        '''
        Mode 1: if source and target belong to the same sentence,
                considered existence of blame
        '''
        s_si, _ = zip(*spos)
        t_si, _ = zip(*tpos)
        common_si = set(s_si) & set(t_si)
        if len(common_si) > 0:
            return 1
        else:
            return 0

    def _mode2(self, spos, tpos):
        '''
        Mode 2: if source and target are within 3 sentences,
                considered existence of blame
        '''
        s_si, _ = zip(*spos)
        t_si, _ = zip(*tpos)
        s_si = sorted(s_si)
        t_si = sorted(t_si)
        mindist = abs(s_si[0] - t_si[0])
        for si in s_si:
            for ti in t_si:
                if abs(si - ti) < mindist:
                    mindist = abs(si - ti)
        if mindist <= 3:
            return 1
        else:
            return 0

    def _mode3(self, spos, tpos, sents):
        '''
        Mode 3: if there is `blame` lexicon in sentences containing
                source or target, considered existence of blame
        '''
        s_si, _ = zip(*spos)
        t_si, _ = zip(*tpos)
        union_s = set(s_si) | set(t_si)
        for si in union_s:
            for lex in self.lexicons:
                if lex in sents[si]:
                    return 1
        return 0
