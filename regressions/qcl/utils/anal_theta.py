import copy
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from ..regressors.StandardQCLRegressor import StandardQCLRegressor

"""
prototype
analyze the effects of theta to prediction
"""


def calc_error(qcl: StandardQCLRegressor, tr_X, tr_y):
    pred_y = qcl.predict(tr_X)
    err = mean_squared_error(pred_y, tr_y)
    return err


def analyze_theta(model: StandardQCLRegressor, tr_X, tr_y):
    best_theta = copy.copy(model.vqe_gates.get_params())
    for i in range(len(theta)):
        err_list = []
        theta_i_list = []
        theta = copy.copy(best_theta)
        for theta_i in np.linspace(-2*np.pi, 2*np.pi, 100):
            theta[i] = theta_i
            model.vqe_gates.set_params(theta)
            err = calc_error(model, tr_X, tr_y)
            err_list.append(err)
            theta_i_list.append(theta_i)

        plt.figure()
        plt.scatter(theta_i_list, err_list)
        plt.plot([best_theta[i], best_theta[i]], [
                 min(err_list)*0, max(err_list)])
