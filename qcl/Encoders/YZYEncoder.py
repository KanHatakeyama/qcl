from qulacs import QuantumCircuit
from typing import List
from ..utils.GPUSetting import QuantumState
import numpy as np
from .StandardEncoder import StandardEncoder
from .prerocess import *

"""
YZY rotation
"""


class YZYEncoder(StandardEncoder):
    def __init__(self, n_qubit: int,
                 scaling_coeff=1.,
                 preprocess_func=three_unity_and_arc_angles):
        super().__init__(n_qubit)
        self.scaling_coeff = scaling_coeff
        self.preprocess_func = preprocess_func

    def _proprocess(self, x: List[float]) -> np.array:
        self.x_dim = len(x)
        return self.preprocess_func(x, self.scaling_coeff)

    def _encode(self, args) -> QuantumState:
        x, y, z = args
        U = QuantumCircuit(self.n_qubit)
        for i in range(self.n_qubit):
            x_id = i % (self.x_dim)
            U.add_RY_gate(i, x[x_id])
            U.add_RZ_gate(i, y[x_id])
            U.add_RY_gate(i, z[x_id])

        return self._calculate_state(U)
