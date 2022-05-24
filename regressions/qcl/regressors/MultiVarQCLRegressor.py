
from qulacs import ParametricQuantumCircuit
from ..gates.CNOTRotationGates import rand_angle
from ..utils.solver import basinhopping_solver_verbose
from ..gates.Observable import ZObservable
import numpy as np
from typing import List
from ..utils.GPUSetting import QuantumState


"""
Prototype regressor class
"""


N_R_GATES = 3


class MultiVarQCLRegressor:
    def __init__(self, n_qubit,
                 x_dim,
                 solver=basinhopping_solver_verbose,
                 observable=None,
                 c_depth=4,
                 scaling_coeff=0.8,
                 ):
        self.n_qubit = n_qubit
        self.x_dim = x_dim
        self.solver = solver
        self.scaling_coeff = scaling_coeff
        self.c_depth = c_depth

        # set observable
        if observable is None:
            self.observable = ZObservable(
                self.n_qubit, coeff=2.0, command="Z 0")
        else:
            self.observable = observable

        self.theta_list = [rand_angle()
                           for _ in range(N_R_GATES*c_depth*n_qubit)]

    def _calculate_state(self, x):
        theta_list = self.theta_list
        n_qubit = self.n_qubit
        x_dim = self.x_dim
        c_depth = self.c_depth

        x = np.array(x)*self.scaling_coeff
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x)

        state = QuantumState(n_qubit)
        circuit = ParametricQuantumCircuit(n_qubit)
        for depth in range(c_depth):
            gate_id = depth % n_qubit
            dim = depth % x_dim
            #print(f"depth {depth} gate {gate_id}, x dimension {dim}")

            # add rotation gates
            circuit.add_parametric_RY_gate(
                gate_id, theta_list[depth*N_R_GATES]*angle_y[dim])
            circuit.add_parametric_RZ_gate(
                gate_id, theta_list[depth*N_R_GATES+1]*angle_z[dim])
            # circuit.add_parametric_RY_gate(
            #    gate_id, theta_list[depth*N_R_GATES+2]*angle_y[dim])
            circuit.add_parametric_RY_gate(
                gate_id, theta_list[depth*N_R_GATES+2]*angle_y[dim])

            # cnot
            if gate_id == n_qubit-1:
                target_id = 0
            else:
                target_id = gate_id+1
            circuit.add_CNOT_gate(gate_id, target_id)

        circuit.update_quantum_state(state)
        self.circuit = circuit
        return state

    def _predict(self, x: List[float]) -> float:
        state = self._calculate_state(x)
        return self.observable(state)

    def predict(self, x_array: np.array):
        if x_array.shape[1] != self.x_dim:
            raise ValueError(
                "dimension of X and x_dim does not match! ", self.x_dim, x_array.shape[1])
        return np.array([self._predict(list(x)) for x in x_array])

    def fit(self, tr_X: np.array, tr_y: np.array):
        tr_y = tr_y.reshape(-1)

        # define loss func
        def cost_func(theta):
            self.theta_list = theta
            y_pred = self.predict(tr_X)
            loss = ((y_pred - tr_y)**2).mean()
            return loss

        result = self.solver(cost_func, self.theta_list)

        self.theta_list = (result.x)
        return self
