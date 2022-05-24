from qulacs import QuantumCircuit
from typing import List
from ..utils.GPUSetting import QuantumState
import numpy as np

"""
encoders
"""


class StandardEncoder:
    """
    simple Rx gate encoder
    """

    def __init__(self, n_qubit: int):
        self.n_qubit = n_qubit

    def _proprocess(self, x: List[float]) -> np.array:
        processed = np.array(x)
        return processed

    def _encode(self, args) -> QuantumState:
        x = args
        U = QuantumCircuit(self.n_qubit)
        x_dim = len(x)

        for i in range(self.n_qubit):
            x_id = i % (x_dim)
            U.add_RX_gate(i, x[x_id])
        return self._calculate_state(U)

    def _calculate_state(self, U: QuantumCircuit) -> QuantumState:
        state = QuantumState(self.n_qubit)
        state.set_zero_state()
        U.update_quantum_state(state)
        return state

    def __call__(self, x: List[float]) -> QuantumState:
        args = self._proprocess(x)
        return self._encode(args)


def get_angle_yz(x: List[float], scaling_coeff: float) -> np.array:
    x = np.array(x)*scaling_coeff
    angle_y = np.arcsin(x)
    angle_z = np.arccos(x)
    return angle_y, angle_z
