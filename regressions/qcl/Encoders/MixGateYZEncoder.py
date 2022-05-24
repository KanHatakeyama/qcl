from qulacs import QuantumCircuit
from typing import List
from ..utils.GPUSetting import QuantumState
import numpy as np
from .StandardEncoder import StandardEncoder
from .prerocess import *

"""
Mix gate YZ rotation
"""


class MixGateYZEncoder(StandardEncoder):
    def __init__(self, n_qubit: int,
                 scaling_coeff=1.,
                 preprocess_func=two_arc_angles):
        super().__init__(n_qubit)
        self.scaling_coeff = scaling_coeff
        self.preprocess_func = preprocess_func

    def _proprocess(self, x: List[float]) -> np.array:
        self.x_dim = len(x)
        return self.preprocess_func(x, self.scaling_coeff)

    def _encode(self, args) -> QuantumState:
        angle_y, angle_z = args
        U = QuantumCircuit(self.n_qubit)
        loop = 0
        for i in range(self.x_dim):
            gate_id = i % (self.n_qubit)
            if gate_id == 0:
                loop += 1

            if loop == 1:
                U.add_RY_gate(gate_id, angle_y[i])
                # print(i,gate_id,loop,"Y")
            elif loop == 2:
                U.add_RZ_gate(gate_id, angle_z[i])
                # print(i,gate_id,loop,"Z")
            else:
                raise ValueError(
                    "too large x dimensions than gates", self.n_qubit, self.x_dim)

        return self._calculate_state(U)
