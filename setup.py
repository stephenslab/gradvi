import sys
import setuptools
from configparser import ConfigParser
from setuptools.command.bdist_egg import bdist_egg

'''
--------------------------
    UTILITIES
--------------------------
'''
def cfg_string_tolist(cfgstr, sep):
    return [x.strip() for x in cfgstr.split(sep) if len(x) > 0]


def cfg_require_version(rlist, cfg):
    minkeys = list(cfg.keys())
    newlist = rlist.copy()
    for i, req in enumerate(rlist):
        minkey = 'min_{}'.format(req)
        maxkey = 'max_{}'.format(req)
        rstrl  = []
        if (minkey in cfg.keys()):
            rstrl.append('>={}'.format(cfg[minkey]))
        if (maxkey in cfg.keys()) and (cfg['release_branch'] == 'True'):
            rstrl.append('<={}'.format(cfg[maxkey]))
        newlist[i] = req + ','.join(rstrl)
    return newlist


class bdist_egg_disabled(bdist_egg):
    '''
    Disabled version of bdist_egg
    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    '''
    def run(self):
        sys.exit("ERROR: aborting implicit building of eggs. Use \"pip install .\" to install from source.")

'''
--------------------------
    SETUP
--------------------------
'''
def setup_package(config_file):

    # all settings are in configs.ini except version number
    config  = ConfigParser(delimiters=['='])
    config.read(config_file)
    cfg     = config['metadata']
    srcdir  = cfg['srcdir']
    pkgname = cfg['name']
    sys.path.append(srcdir)
    sys.path.append('{:s}/{:s}'.format(srcdir, pkgname))
    from version import __version__

    setup_requires     = cfg_string_tolist(cfg['setup_requires'], ',')
    install_requires   = cfg_string_tolist(cfg['install_requires'], ',')

    install_requires_v = cfg_require_version(install_requires, cfg)
    req_py = cfg_require_version(['python'], cfg)[0]
    req_py = req_py[6:]

    console_entry_point = ['{:s} = {:s}.main:main'.format(pkgname, pkgname)]

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    cmdclass = {
        'bdist_egg': bdist_egg if 'bdist_egg' in sys.argv else bdist_egg_disabled,
    }

    metadata = dict(
        name             = pkgname,
        version          = __version__,
        author           = cfg['author'],
        author_email     = cfg['author_email'],
        description      = cfg['description'],
        long_description = long_description,
        long_description_content_type = "text/markdown",
        license          = cfg['license'],
        url              = cfg['main_url'],
        download_url     = cfg['download_url'],
        project_urls     = {
            "Bug Tracker":   cfg['bug_tracker_url'],
            "Documentation": cfg['docs_url'],
            "Source Code" :  cfg['source_code_url'],
        },
        classifiers      = cfg_string_tolist(cfg['classifiers'], '\n'),
        packages         = setuptools.find_packages(where = srcdir),
        package_dir      = {"": srcdir},
        entry_points     = {'console_scripts': console_entry_point},
        cmdclass         = cmdclass,
        python_requires  = req_py,
        setup_requires   = setup_requires,
        install_requires = install_requires,
        keywords         = cfg_string_tolist(cfg['keywords'], ','),
    )

    setuptools.setup(**metadata)

if __name__ == '__main__':
    setup_package('configs.ini')
