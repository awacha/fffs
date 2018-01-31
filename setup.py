#!/usb/bin/env python
import os
import sys

from Cython.Build import cythonize
from numpy.lib import get_include
from setuptools import setup, find_packages, Extension


def getresourcefiles():
    print('Generating resource list', flush=True)
    reslist = []
    for directory, subdirs, files in os.walk(os.path.join('src','fffs','resource')):
        reslist.extend([os.path.join(directory, f).split(os.path.sep, 1)[1] for f in files])
    print('Generated resource list:\n  ' + '\n  '.join(x for x in reslist) + '\n', flush=True)
    return reslist


extensions = []

for directory, subdirs, files in os.walk(os.path.join('src', 'fffs')):
    for f in files:
        if f.endswith('.pyx'):
            pathelements=os.path.normpath(os.path.join(directory,f[:-4])).split(os.path.sep)
            pathelements=pathelements[1:] # remove the first 'src' term
            extensions.append(Extension(
                ".".join(pathelements),
                [os.path.join(directory, f)],
                include_dirs = [get_include()],
#                libraries = []
            ))

setup(name='fffs', author='Andras Wacha',
      author_email='awacha@gmail.com', url='http://github.com/awacha/fffs',
      description='Fitting Functions for SAXS',
      package_dir = {'':'src'},
      packages=find_packages('src'),
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'cython'],
      #      cmdclass = {'build_ext': build_ext},
      ext_modules=cythonize(extensions),
      install_requires=['numpy>=1.11.1', 'scipy>=0.18.0', 'cython'],
      keywords="saxs sans sas small-angle scattering x-ray fitting least squares",
      license="",
      package_data={'': getresourcefiles()},
      #      include_package_data=True,
      zip_safe=False,
      )
