#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
This script will write a timestamp to a time rotating log file. The purpose is to use this as a test bed for
implementing the rotating log file handler in other projects.
"""

import os
import argparse
import sys


class RunDBCheck(object):

    VERSION = 'RunDBCheck.py 03Aug2017 J.Arndt - Sondrestrom Radar'
    base_directory = "RADAR_DATA"

    def __init__(self):
        self.debug = False

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Radar Run Database Utilities')
        # utility_group = parser.add_argument_group('utilities')
        option_group = parser.add_argument_group('options')

        # utility_group.add_argument('-s', '--get-sensor-status', help='Get sensor status', action='store_true',
        #                            dest='get_sensor_status')
        # utility_group.add_argument('--add-sensor-availability', nargs=4,
        #                            metavar=('START', 'END', 'STATUS', 'NOTES'),
        #                            dest='add_sensor_availability', help='Add an availability entry for sensor')
        #
        # option_group.add_argument('--local', action='store_true', help='Print local time instead of UTC')
        option_group.add_argument('-v', '--version', action='store_true')

        args = parser.parse_args(sys.argv[1:])

        try:
            if args.version:
                self.version()

        except Exception as e:
            print(e.message)

    @staticmethod
    def version():
        """Print the program version """
        print(RunDBCheck.VERSION)
        sys.exit(0)

    @staticmethod
    def list_directories(self):
        return (dI for dI in os.listdir(self.base_directory) if os.path.isdir(os.path.join(self.base_directory, dI)))

if __name__ == '__main__':
    c = RunDBCheck()
    c.parse_args()
