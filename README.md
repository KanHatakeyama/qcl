# Quantum circuit learning (QCL) for Regression

# About
- Source codes and classes to conduct QCL for regression tasks

# Quick links
- Start
  1. [Simple regression for 1d data](QCLRegressor.ipynb)
  2. [Visualization of quantum states](Quantum_state_visualizer.ipynb)
- Details (results discussed in paper)
    - One-dimensional dataset
      1. [Optimization of circuit structure](regressions/1D_optimize_model.ipynb)
      2. [Calculation time](regressions/1D_calc_time.ipynb)
      3. [Comparison with conventional models](regressions/1D_comparison.ipynb)
      4. [Regression by actual IBM quantum machines](regressions/1D_Qiskit.ipynb)
    - Chemical dataset
        - [Predicting chemical properties](regressions/Chem_calc.ipynb)
# Cite
- Manuscript in preparation

# Related info 
- [QCL paper by Mitarai et al (Phys. Rev. A 2018)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.032309)
- [Quantum Native Dojo](https://dojo.qulacs.org/)

# Main modules
```
python==3.9.12
sympy==1.10.1
scipy==1.8.0
qulacs==0.3.0
qulacsvis==0.2.2
qiskit==0.36.1
scikit-learn==1.0.2
keras==2.9.0
tensorflow==2.9.0
```

# Author
- Kan Hatakeyama-Sato
- Waseda University
- https://kanhatakeyama.github.io/
