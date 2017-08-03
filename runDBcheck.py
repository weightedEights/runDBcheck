#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
This script will write a timestamp to a time rotating log file. The purpose is to use this as a test bed for
implementing the rotating log file handler in other projects.
"""


class RunDBCheck(object):

    def __init__(self):
        self.debug = False

    def parse_args(self):
        pass


if __name__ == '__main__':
    c = RunDBCheck()
    c.parse_args()
