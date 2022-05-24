
from ..Encoders.StandardEncoder import StandardEncoder
from ..gates.CNOTRotationGates import CNOTRotationGates
from ..gates.Observable import ZObservable
from ..utils.solver import basinhopping_solver, basinhopping_solver_verbose
from typing import List
from ..utils.GPUSetting import QuantumState
import numpy as np

"""
standard QCL regressor
"""


class StandardQCLRegressor:
    def __init__(self, n_qubit: int,
                 x_dim: int,
                 encoder=None,
                 solver=basinhopping_solver_verbose,
                 observable=None,
                 vqe_gates=None,
                 sigmoid_x=False,
                 logit_y_threshold=None,
                 ):
        self.n_qubit = n_qubit
        self.x_dim = x_dim
        self.solver = solver
        self.sigmoid_x = sigmoid_x
        self.logit_y_threshold = logit_y_threshold

        # set encoder
        if encoder is None:
            self.encoder = StandardEncoder(self.n_qubit)
        else:
            self.encoder = encoder

        # set U gate
        if vqe_gates is None:
            self.vqe_gates = CNOTRotationGates(self.n_qubit, depth=2)
        else:
            self.vqe_gates = vqe_gates

        # set observable
        if observable is None:
            self.observable = ZObservable(
                self.n_qubit, coeff=2.0, command="Z 0")
        else:
            self.observable = observable

    def _calc_entangled_states(self, state: QuantumState) -> QuantumState:
        self.vqe_gates(state)
        return state

    def _predict(self, x: List[float]) -> float:
        if self.sigmoid_x:
            x = 1/(1+np.exp(-np.array(x)))
        state = self.encoder(x)
        state = self._calc_entangled_states(state)
        y = self.observable(state)

        if self.logit_y_threshold:
            # avoid nan
            v = abs(self.logit_y_threshold)
            y = y/2+1/2
            if y > v:
                y = v
            elif y < 1-v:
                y = 1-v
            # print(y)
            y = np.log(y/(1-y))
            if y != y:
                y = 0
        return y

    def predict(self, x_array: np.array):
        if x_array.shape[1] != self.x_dim:
            raise ValueError(
                "dimension of X and x_dim does not match! ", self.x_dim, x_array.shape[1])
        return np.array([self._predict(list(x)) for x in x_array])

    def fit(self, tr_X: np.array, tr_y: np.array):
        tr_y = tr_y.reshape(-1)

        # define loss func
        def cost_func(theta):
            self.vqe_gates.set_params(theta)
            y_pred = self.predict(tr_X)
            loss = ((y_pred - tr_y)**2).mean()
            return loss

        result = self.solver(cost_func, self.vqe_gates.get_params())

        self.vqe_gates.set_params(result.x)
        return self
