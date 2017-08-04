#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
This script will contain utilities for checking the continuity of the radar runs database. Primarily that the file
numbers are continuous. If there is a discontinuity, highlight the boundary experiments. 

Usage:

counterLogParse.py [-v] [-h] [-p] [-r <logFileRangeStart> <logFileRangeStop>]

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


class RunDBCheck(object):

    VERSION = 'RunDBCheck.py - 03Aug2017 J.Arndt - Sondrestrom Radar'
    base_directory = "RADAR_DATA"

    def __init__(self):
        self.debug = False

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Radar Run Database Utilities')

        parser.add_argument('-v', '--version', action='store_true', help='Print version')
        parser.add_argument('-c', '--continuity', action='store', dest='check_continuity', metavar='PATH',
                            help='Check file number continuity within RUNS.DB. Path in posix format: "./directory/"')

        # utility_group.add_argument('-s', '--get-sensor-status', help='Get sensor status', action='store_true',
        #                            dest='get_sensor_status')
        # utility_group.add_argument('--add-sensor-availability', nargs=4,
        #                            metavar=('START', 'END', 'STATUS', 'NOTES'),
        #                            dest='add_sensor_availability', help='Add an availability entry for sensor')

        args = parser.parse_args(sys.argv[1:])

        try:
            if args.version:
                self.version()
            elif args.check_continuity:
                self.check_continuity(args.check_continuity)

        except Exception as e:
            print(e.message)

    @staticmethod
    def version():
        """Print the program version """
        print(RunDBCheck.VERSION)
        sys.exit(0)

    def check_continuity(self, path):
        """Check RUNS.DB file Number continuity"""
        print("Will check directory: {}".format(path))
        if not os.path.exists(path):
            raise Exception('ERROR check_continuity: incorrect path')
        else:
            print("Path exists.")

    @staticmethod
    def list_directories(self):
        return (dI for dI in os.listdir(self.base_directory) if os.path.isdir(os.path.join(self.base_directory, dI)))

if __name__ == '__main__':
    c = RunDBCheck()
    c.parse_args()
