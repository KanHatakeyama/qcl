from setuptools import setup, find_packages
import sys

sys.path.append('./regressions/qcl')

setup(name='QCLRegressor',
      version='0.0.1',
      description='QCLRegressor',
      long_description="README",
      author='Kan Hatakeyama',
      license="MIT",
      packages=find_packages(),
      # packages=["Encoders", "gates", "Qiskit",
      #          "regressors", "testing", "utils"],
      )
