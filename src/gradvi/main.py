
"""
Main command line options to run GradVI
"""

import numpy as np
import argparse
import sys
import unittest

from .utils.logs import MyLogger
from .utils import project
from .tests.run import run_unittests

mlogger = MyLogger(__name__)

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
    res = parser.parse_args()
    return res


def do_task(opts):
    print ("This is not implemented yet.")
    return


def main():

    opts = parse_args()
    if opts.test:
        run_unittests()
    elif opts.version:
        print ("GradVI version {:s}".format(project.version()))
    elif opts.do_lbfgsb:
        do_task(opts)

if __name__ == "__main__":
    main()
