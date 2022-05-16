import os
import logging

def version():
    ## Get the version from version.py without importing the package
    vfn = os.path.join(os.path.dirname(__file__), '../version.py')
    exec(compile(open(vfn).read(), vfn, 'exec'))
    res = locals()['__version__']
    return res


def get_name():
    namestr = __name__
    return namestr.strip().split('.')[0].strip()


def logging_level():
    level = logging.WARN
    #if lstring is not None:
    #    lstr = lstring.lower()
    #    if lstr == 'debug':
    #        level = logging.DEBUG 
    #    elif lstr == 'info':
    #        level = logging.INFO
    #    elif lstr == 'warn':
    #        level = logging.WARN
    return level


def logging_format():
    return "%(asctime)s | %(name)-40s | %(levelname)-7s | %(message)s"


def logging_file():
    return None


def libprefix():
    return "libmrashpen"


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
