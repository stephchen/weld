import os
import platform
import shutil
import subprocess
import sys

from setuptools import setup, Distribution
import setuptools.command.build_ext as _build_ext
from setuptools.command.install import install

class Install(install):
    def run(self):
        install.run(self)
        python_executable = sys.executable

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(name='weldsklearn',
      version='0.0.1',
      packages=['weldsklearn'],
      cmdclass={"install": Install},
      distclass=BinaryDistribution,
      install_requires=['pyweld'])
