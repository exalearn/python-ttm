from numpy.distutils.core import Extension
from numpy.distutils.core import setup
from setuptools import find_packages
import os

# Write out the extension
ext = Extension(name='ttm.flib',
                sources=[os.path.join('ttm', 'ttm_from_f2py.f90')],
                extra_f90_compile_args=['-O2', '-fPIC'],
                include_dirs=['pot_ttm'],
                library_dirs=['pot_ttm'],
                extra_objects=[os.path.abspath('./pot_ttm/pot_ttm.a')],
                extra_link_args=['-fPIC', '-shared'])

setup(name='ttm', packages=find_packages(), ext_modules=[ext], version='0.0.1')
