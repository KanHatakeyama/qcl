from setuptools import setup, find_packages
import sys

sys.path.append('./qcl')

setup(name='QCLRegressor',
      version='2022.5.25',
      description='QCLRegressor',
      long_description="README",
      author='Kan Hatakeyama',
      license="MIT",
      packages=find_packages(),
      )
