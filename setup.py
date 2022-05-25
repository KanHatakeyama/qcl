from setuptools import setup, find_packages
import sys

sys.path.append('./regressions/qcl')

setup(name='QCLRegressor',
      version='0.0.1',
      description='QCLRegressor',
      long_description="README",
      author='Kan Hatakeyama',
      license="MIT",
      # packages=find_packages(),
      packages=[
          "QCLRegressor.Encoders",
          "QCLRegressor.gates",
          "QCLRegressor.Qiskit",
          "QCLRegressor.regressors",
          "QCLRegressor.testing",
          "QCLRegressor.utils",
      ],

      package_dir={
          "QCLRegressor.Encoders": "regressions/qcl/Encoders",
          "QCLRegressor.gates": "regressions/qcl/gates",
          "QCLRegressor.Qiskit": "regressions/qcl/Qiskit",
          "QCLRegressor.regressors": "regressions/qcl/regressors",
          "QCLRegressor.testing": "regressions/qcl/testing",
          "QCLRegressor.utils": "regressions/qcl/utils"
      },
      )
"""
packages=['regressions/qcl/Encoders',
            'regressions/qcl/gates',
            'regressions/qcl/Qiskit',
            'regressions/qcl/regressors',
            'regressions/qcl/testing',
            'regressions/qcl/utils'],
"""
