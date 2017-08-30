#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
This script will contain utilities for checking the continuity of the radar runs database. Primarily that the file
numbers are continuous. If there is a discontinuity, highlight the boundary experiments. 

Usage:

runDBcheck.py [-v] [-h] [-p] [-r <logFileRangeStart> <logFileRangeStop>]

[-v]
    Print script version
[-h]
    Print this usage
[-c] <path-to-RUNS.DB>
    Check file number continuity within RUNS.DB. Path should be in posix format: "./directory/"
[-r <path-to-experiments> <logFileRangeStart> <logFileRangeStop>]
    Check for experiment.log.html file in the standard experiment directory structure. Path must be the parent
    directory to the experiment directory and in posix format. 
"""

import os
import argparse
import sys
from collections import OrderedDict


class RunDBCheck(object):

    VERSION = 'RunDBCheck.py - 03Aug2017 J.Arndt - Sondrestrom Radar'
    RUNS_DB_FILENAME = "sampleRUNS.DB"

    def __init__(self):
        self.debug = False

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Radar Run Database Utilities')

        parser.add_argument('-v', '--version', action='store_true', help='Print version')
        parser.add_argument('-c', '--continuity', action='store', dest='check_continuity', metavar='PATH',
                            help='Check file number continuity within RUNS.DB.')
        parser.add_argument('-r', '--experiment-range', nargs='+', metavar=('PATH', 'START'),
                            default='allOfThem', dest='log_html_check',
                            help='Check range of experiments for ./log/log.html. If no starting directory is specified,'
                                 ' will check whole parent directory. If no stopping directory, will check from start'
                                 ' to end.')

        args = parser.parse_args(sys.argv[1:])

        try:
            if args.version:
                self.version()
            elif args.check_continuity:
                self.check_continuity(args.check_continuity)
            elif args.log_html_check:
                self.log_html_check(args.log_html_check)

        except Exception as e:
            print(e.message)

    @staticmethod
    def version():
        """Print the program version """
        print(RunDBCheck.VERSION)
        sys.exit(0)

    @staticmethod
    def runs_db_reader(path):
        print("done.")
        db_actual = OrderedDict()
        with open(path) as fin:
            for line in fin:
                # throw away empty lines
                if line != '\n':
                    # check if it should be a new key, else a new value for the previous key
                    if line[0] == '[':
                        new_key = line.rstrip()
                        db_actual[new_key] = []
                    else:
                        db_actual[new_key].append(line.rstrip())

        for k, v in db_actual.items():
            print(k, v)

    @staticmethod
    def check_continuity(path):
        """Check RUNS.DB file Number continuity"""
        print("Will check directory: {}".format(path))

        full_runs_path = os.path.join(path, RunDBCheck.RUNS_DB_FILENAME)

        if not os.path.exists(path):
            raise Exception('ERROR: check_continuity: incorrect path')

        if os.path.exists(full_runs_path) and os.path.isfile(full_runs_path):
            print("RUNS.DB found. Calling reader..")
            RunDBCheck.runs_db_reader(full_runs_path)

        else:
            raise Exception('ERROR: check_continuity: RUNS.DB not found.')

    @staticmethod
    def log_html_check(args):
        """Check specified experiments for log.html file
        The default experiment directory structure will be assumed..
        <path>/experiment/Log/experiment.log.html
        """
        # define log file suffix
        suffix = ".log.html"

        # check if user-supplied path exists
        path = args[0]
        if not os.path.exists(path):
            raise Exception('ERROR: path does not exist')

        print("Checking directory: {} ...".format(path))

        # check if START experiment is supplied
        try:
            start = args[1]
            if os.path.exists(os.path.join(path, start)):
                # print("Starting experiment: ", start)
                pass
            else:
                raise Exception('ERROR: START experiment does not exist')
        except IndexError:
            pass

        # check if STOP experiment is supplied
        try:
            stop = args[2]
            if os.path.exists(os.path.join(path, stop)):
                # print("Ending experiment: ", stop)
                pass
            else:
                raise Exception('ERROR: STOP experiment does not exist')
        except IndexError:
            pass

        # create list of all experiment directories
        list_dir = [d for d in os.listdir(path)]

        # check if the log file exists for each experiment directory. if not, flag it.
        for exp in list_dir:
            if os.path.exists(os.path.join(path, exp, "Log", exp + suffix)):
                print("Experiment {} log exists.".format(exp))
            else:
                print("Experiment {} log IS MISSING. ---".format(exp))

if __name__ == '__main__':
    c = RunDBCheck()
    c.parse_args()
