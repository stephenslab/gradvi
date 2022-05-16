
"""
Main command line options to run GradVI
"""

import numpy as np
import argparse
import sys
import unittest
import logging

from .utils.logs import CustomLogger
from .utils import project
from .tests.run import run_unittests

mlogger = CustomLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='GradVI: gradient descent methods for mean field variational inference')
    parser.add_argument('--test',
                        dest = 'test',
                        action = 'store_true',
                        help = 'Perform unit tests')
    parser.add_argument('--version',
                        dest = 'version',
                        action = 'store_true',
                        help = 'Print version number')
    parser.add_argument('--verbose',
                        dest = 'verbose',
                        action = 'store_true',
                        help = 'Print information while running')
    parser.add_argument('--vverbose',
                        dest = 'vverbose',
                        action = 'store_true',
                        help = 'Print more information while running')
    res = parser.parse_args()
    return res


def do_task(opts):
    print ("This is not implemented yet.")
    return


def main():
    opts = parse_args()
    log_level = logging.INFO if opts.verbose else None
    log_level = logging.DEBUG if opts.vverbose else log_level
    mlogger.set_loglevel(log_level)
    mlogger.override_global_default_loglevel(log_level)
    if opts.test:
        mlogger.debug("Calling logger from main")
        run_unittests()
    elif opts.version:
        print ("GradVI version {:s}".format(project.version()))
    elif opts.do_lbfgsb:
        do_task(opts)

if __name__ == "__main__":
    main()
