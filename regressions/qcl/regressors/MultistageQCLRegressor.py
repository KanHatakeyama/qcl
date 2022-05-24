import numpy as np
from qulacs import ParametricQuantumCircuit
from ..utils.solver import basinhopping_solver_verbose
from ..utils.GPUSetting import QuantumState
from ..utils.funcs import rand_angle
from .MultiVarQCLRegressor import MultiVarQCLRegressor

"""
Prototype regressor class
"""


class MultistageQCLRegressor(MultiVarQCLRegressor):
    def __init__(self, n_qubit,
                 x_dim,
                 solver=basinhopping_solver_verbose,
                 observable=None,
                 c_depth=1,
                 scaling_coeff=1.0,
                 ):
        self.n_qubit = n_qubit
        self.x_dim = x_dim
        self.solver = solver
        self.scaling_coeff = scaling_coeff
        self.c_depth = c_depth

        if n_qubit*2 != x_dim:
            raise ValueError("n_qubit must match x_dim/2", n_qubit, x_dim)

        super().__init__(n_qubit, x_dim, solver, observable, c_depth, scaling_coeff)

        self.theta_list = [rand_angle()
                           for _ in range(3*n_qubit*2+3*c_depth)]

    def _calculate_state(self, x):
        n_qubit = self.n_qubit

        x = np.array(x)*self.scaling_coeff
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x)

        state = QuantumState(n_qubit)
        self.circuit = ParametricQuantumCircuit(n_qubit)

        theta_id = 0

        def add_interactions(theta_id):
            # add CNOT gates with neighboring bits
            for i in range(self.n_qubit-1):
                #print(i, i+1)
                self.circuit.add_CNOT_gate(i, i+1)

            # add rotation gates
            for i in range(self.n_qubit):
                self.circuit.add_parametric_RX_gate(
                    i, self.theta_list[theta_id])
                theta_id += 1
                self.circuit.add_parametric_RZ_gate(
                    i, self.theta_list[theta_id])
                theta_id += 1
                self.circuit.add_parametric_RX_gate(
                    i, self.theta_list[theta_id])
                theta_id += 1

            return theta_id

        # embed first half part of X
        for qubit_id in range(n_qubit):
            x_id = qubit_id
            self.circuit.add_RY_gate(qubit_id, angle_y[x_id])
            self.circuit.add_RZ_gate(qubit_id, angle_z[x_id])

        theta_id = add_interactions(theta_id)

        # embed last half part of X
        for qubit_id in range(n_qubit):
            x_id = qubit_id+n_qubit
            self.circuit.add_RY_gate(qubit_id, angle_y[x_id])
            self.circuit.add_RZ_gate(qubit_id, angle_z[x_id])

        for d in range(self.c_depth):
            theta_id = add_interactions(theta_id)

        self.circuit.update_quantum_state(state)
        return state
