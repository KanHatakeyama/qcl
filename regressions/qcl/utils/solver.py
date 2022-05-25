import numpy as np
from scipy.optimize import minimize, basinhopping

"""
solver functions for QCL models
"""


def nelder_mead_solver(cost_func, theta_0):

    bounds = ([[-np.pi, np.pi] for _ in range(len(theta_0))])
    result = minimize(cost_func,
                      theta_0,
                      bounds=bounds,
                      method='Nelder-Mead',)
    return result


def basinhopping_solver_verbose(cost_func, theta_0, n_iter=0):
    return basinhopping_solver(cost_func, theta_0, n_iter=n_iter, verbose=True)


def basinhopping_solver(cost_func, theta_0, n_iter=50, verbose=False, xmax=np.pi, xmin=-np.pi):
    # print(theta_0)

    def callback_print(a, loss, c):
        print(f"loss {loss:.15f}")

    if verbose:
        callback = callback_print
    else:
        callback = None

    minimizer_kwargs = {"method": "BFGS"}
    result = basinhopping(cost_func, theta_0, minimizer_kwargs=minimizer_kwargs,
                          accept_test=BasinBounds(xmax=xmax, xmin=xmin),
                          niter=n_iter,
                          callback=callback,
                          )
    return result


class BasinBounds:
    def __init__(self, xmax=np.pi, xmin=-np.pi):
        self.xmax = xmax
        self.xmin = xmin

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin
