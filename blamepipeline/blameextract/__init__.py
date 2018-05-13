#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2018-05-09 11:12:33
# @Last Modified by:  Shuailong
# @Last Modified time: 2018-05-09 11:17:34

from .. import DATA_DIR

DEFAULTS = {

}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .model import BlameExtractor
