from qulacs import QuantumCircuit
from typing import List
from ..utils.GPUSetting import QuantumState
import numpy as np
from .StandardEncoder import StandardEncoder
from .prerocess import *

"""
X,Y,or Z rotation
"""


class WEncoder(StandardEncoder):
    def __init__(self, n_qubit: int,
                 scaling_coeff=1.,
                 preprocess_func=one_unity,
                 angle_mode="x"):
        super().__init__(n_qubit)
        self.scaling_coeff = scaling_coeff
        self.preprocess_func = preprocess_func
        self.angle_mode = "x"

        if angle_mode not in ["x", "y", "z"]:
            raise ValueError("angle modes must be x, y, or z", angle_mode)

    def _proprocess(self, x: List[float]) -> np.array:
        self.x_dim = len(x)
        return self.preprocess_func(x, self.scaling_coeff)

    def _encode(self, args) -> QuantumState:
        y = args
        U = QuantumCircuit(self.n_qubit)
        for i in range(self.n_qubit):
            x_id = i % (self.x_dim)

            if self.angle_mode == "x":
                U.add_RX_gate(i, y[x_id])
            elif self.angle_mode == "y":
                U.add_RY_gate(i, y[x_id])
            elif self.angle_mode == "x":
                U.add_RZ_gate(i, y[x_id])
        return self._calculate_state(U)
