import sys
import os
import subprocess
import setuptools
import copy
import textwrap
from configparser import ConfigParser

from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.install   import install
#from setuptools.command.build_ext import build_ext
from setuptools                   import Extension
from distutils                    import log
from numpy.distutils.core import setup as np_setup
from numpy.distutils.core import Extension as np_Extension
from numpy.distutils.command.build_ext import build_ext as np_build_ext

'''
--------------------------
    SETTINGS
--------------------------
All settings are in configs.ini except version number.
'''
config = ConfigParser(delimiters=['='])
config.read('configs.ini')
cfg = config['metadata']
sys.path.append('src')
sys.path.append('src/{:s}'.format(cfg['name']))
from version import __version__

def cfg_string_tolist(key, sep):
    return [x.strip() for x in cfg[key].split(sep) if len(x) > 0]

'''
--------------------------
    LINEAR ALGEBRA
--------------------------
We want to get information about the linear algebra library
installed in the system.
Hack: piggyback on the numpy setuptools, 
      copy system_info.py and cpuinfo.py
It provides information about various resources (libraries, library directories,
include directories, etc.) in the system. Usage:
    info_dict = get_info(<name>)
  where <name> is a string 'atlas','x11','fftw','lapack','blas',
  'lapack_src', 'blas_src', etc. For a complete list of allowed names,
  see the definition of get_info() function in src/system_info.py
  Returned info_dict is a dictionary which is compatible with
  distutils.setup keyword arguments. If info_dict == {}, then the
  asked resource is not available (system_info could not find it).
  Several *_info classes specify an environment variable to specify
  the locations of software. When setting the corresponding environment
  variable to 'None' then the software will be ignored, even when it
  is available in system.

WARN: Numpy is not installed by default in a Python environment.

We have to make sure `numpy` is installed before the setup.
Possible ways of doing this:
  - define `numpy` in 'setup_requires' (instead of 'install_requires') of setuptools.setup()
  - `pyproject.toml`.

NOTE: I originally used get_info directly from numpy
    from numpy.distutils.system_info import get_info
However, the info is incorrect for AMD / MKL, 
hence I copied the files from numpy and used a hack to 
include Intel MKL on AMD machines.
(see line 1186 in src/system_info.py)
We can later move to numpy distribution of system_info
when the bug is fixed.

Obtain information about system Lapack.
The default order for the libraries are:
  - MKL
  - OpenBLAS
  - libFLAME
  - ATLAS
  - LAPACK (NetLIB)
'''
def numpy_get_lapack_info():
    from system_info import get_info
    return get_info('lapack_opt', 0)


def numpy_get_f2py_info():
    from system_info import get_info
    return get_info('f2py', 0)

def numpy_get_blas_info():
    from system_info import get_info
    return get_info('blas', 0)


'''
--------------------------
    C LIBRARIES
--------------------------
The sources and target shared libraries are defined in the config file.
Example:
  clib_path = src/cpydemo/clibs
  clib_sources = 
    sum :: sum.c
    diff :: diff.c
LAPACK installation of the system is obtained from numpy.distutils.system_info (see above)
and these sources are combined to obtain a list of extension modules
'''

def cfg_lib_todict(libs, lib_dir, libprefix):
    ''' 
    Reads cfg input format of libs
    libs is a list with strings, formatted as
        name :: filea.ext, fileb.ext, ...
    Returns dict {name: list<source_files>}
    '''
    lib_dict = dict()
    for lib in libs:
        strsplit = lib.split('::')
        libname = '{:s}_{:s}'.format(libprefix, strsplit[0].strip())
        libsources = [x.strip() for x in strsplit[1].split(',')]
        lib_dict[libname] = [os.path.join(lib_dir, x) for x in libsources]
    return lib_dict


def compile_extension_dict (name, sources, extra_libraries, extra_compile_args, **kw):

    def dict_append(d, **kws):
        for k, v in kws.items():
            if k in d:
                ov = d[k]
                if isinstance(ov, str):
                    d[k] = v
                else:
                    d[k].extend(v)
            else:
                d[k] = v

    ext_args = copy.copy(kw)
    ext_args['name'] = name
    ext_args['sources'] = sources

    if 'extra_info' in ext_args:
        extra_info = ext_args['extra_info']
        del ext_args['extra_info']
        if isinstance(extra_info, dict):
            extra_info = [extra_info]
        for info in extra_info:
            dict_append(ext_args, **info)

    # Add extra libraries / compile_args 
    libraries = ext_args.get('libraries', [])
    ext_args['libraries'] = libraries + extra_libraries
    ext_args['extra_compile_args'] = extra_compile_args
    #
    return np_Extension(**ext_args)
    #return ext_args


def ext_modules():
    # Current working directory
    cwd         = os.path.abspath(os.path.dirname(__file__))
    # Prefix to be added to shared libraries
    libprefix   = 'lib{:s}'.format(cfg['name'])
    '''
    Compile the C ext_modules
    '''
    clib_dir    = os.path.join(cwd, cfg['clib_path'])
    clib_cfgs   = cfg_string_tolist('clib_sources', '\n')
    cmodules    = [] 
    if len(clib_cfgs) > 0:
        lapack_info = numpy_get_lapack_info()
        print(lapack_info)
        # 
        extra_libraries = []
        extra_compile_args = ['-O3', '-Werror=implicit-function-declaration']
        #
        clibs       = cfg_lib_todict(clib_cfgs, clib_dir, libprefix)
        cmodules    = [compile_extension_dict(k, v, extra_libraries, extra_compile_args,
                          **dict(extra_info = lapack_info)) \
                          for k, v in clibs.items()]
    '''
    Compile the Fortran ext_modules
    '''
    flib_dir    = os.path.join(cwd, cfg['flib_path'])
    flib_cfgs   = cfg_string_tolist('flib_sources', '\n')
    fmodules    = []
    if len(flib_cfgs) > 0:
        f2py_info   = numpy_get_f2py_info()
        blas_info   = numpy_get_blas_info()
        extra_libraries = []
        extra_compile_args = ['-O3', '-Wall', '-fbounds-check', '-g', '-Wno-uninitialized', '-fno-automatic', '-ffast-math']
        #
        flibs       = cfg_lib_todict(flib_cfgs, flib_dir, libprefix)
        #fmodules    = [compile_extension_dict(k, v, extra_libraries, extra_compile_args, \
        #                  **dict(extra_info = blas_info)) \
        #                  for k, v in flibs.items()]
        fmodules    = [compile_extension_dict(k, v, extra_libraries, extra_compile_args)  \
                           for k, v in flibs.items()]
    return cmodules + fmodules

'''
--------------------------
    CUSTOM OVERRIDES
--------------------------
Example custom overrides,
which can be useful to modify the build process.
'''
class custom_build_ext(np_build_ext):

    def build_extensions(self):
        # self.compiler - the system compiler identified by setuptools / distutils
        # print(vars(self.compiler))
        # for compiler_arg in vars(self.compiler):
        #     print (compiler_arg)
        #     print (getattr(self.compiler, compiler_arg))

        #log.info("Building C modules")
        for ext in self.extensions:
            log.info( f"Building {ext.name}" )
            #print (dir(ext))
            #property_list = ['define_macros', 'depends', 'export_symbols', 'extra_compile_args', 
            #                 'extra_link_args', 'extra_objects', 'include_dirs', 'libraries', 
            #                 'library_dirs', 'runtime_library_dirs', 'sources', 'swig_opts', 
            #                 'undef_macros']
            #log.info("Name: ", ext.name)
            #log.info("Language: " , ext.language)
            #log.info("py_limited_api: ", ext.py_limited_api)
            #for info in property_list:
            #    log.info(f"{info}: " + ", ".join(getattr(ext, info)))
        np_build_ext.build_extensions(self)


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
def setup_package():

    if cfg['release_branch'] == 'True':
        req_np = 'numpy>={},<{}'.format(cfg['min_numpy'], cfg['max_numpy'])
        req_py = '>={},<{}'.format(cfg['min_python'], cfg['max_python'])
    else:
        req_np = 'numpy>={}'.format(cfg['min_numpy'])
        req_py = '>={}'.format(cfg['min_python'])

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    cmdclass = {
        'bdist_egg': bdist_egg if 'bdist_egg' in sys.argv else bdist_egg_disabled,
        'build_ext': custom_build_ext,
    }


    #print(numpy_get_lapack_info())

    #cfg_str_keys  = 'name author author_email description copyright license'.split()
    #cfg_url_keys  = 'main_url download_url git_url docs_url source_code_url bug_tracker_url'.split()
    #cfg_list_keys = 'keywords setup_requires install_requires contributiors'.split()

    metadata = dict(
        name             = cfg['name'],
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
        classifiers      = cfg_string_tolist('classifiers', '\n'),
        packages         = setuptools.find_packages(where = "src"),
        package_dir      = {"": "src"},
        entry_points     = {'console_scripts': ['gradvi = gradvi.main:main']},
        #ext_modules      = [ext1],
        ext_modules      = ext_modules(),
        cmdclass         = cmdclass,
        #setup_requires   = cfg_string_tolist('setup_requires', ','),
        python_requires  = req_py,
        install_requires = [req_np] + cfg_string_tolist('install_requires', ','),
        keywords         = cfg_string_tolist('keywords', ','),
    )

    #setuptools.setup(**metadata)
    np_setup(**metadata)


if __name__ == '__main__':
    setup_package()
