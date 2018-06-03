#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .. import DATA_DIR

DEFAULTS = {

}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .model import BlameExtractor
