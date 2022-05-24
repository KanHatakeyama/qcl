import numpy as np
from qulacs import ParametricQuantumCircuit
from typing import List
#from qulacs import QuantumState
from ..utils.GPUSetting import QuantumState
from ..utils.funcs import rand_angle


class CNOTRotationGates:
    def __init__(self,
                 n_qubit: int,
                 depth=2,
                 init_gates=True):
        self.n_qubit = n_qubit
        self.depth = depth
        self.circuit = ParametricQuantumCircuit(n_qubit)

        if init_gates:
            self._initialize_gates()

    def _initialize_gates(self):
        for d in range(self.depth):

            # add CNOT gates to neighboring bits
            if self.n_qubit > 1:
                for i in (range(self.n_qubit)):
                    if i < self.n_qubit-1:
                        self.circuit.add_CNOT_gate(i, i+1)
                    elif self.n_qubit > 2:
                        self.circuit.add_CNOT_gate(0, i)

            # add rotation gates
            for i in range(self.n_qubit):
                # If we observe the qubit of i=0, other qubit rotation in the last depth does not affect the prediction
                if d == self.depth-1 and i > 0:
                    break

                self.circuit.add_parametric_RX_gate(i, rand_angle())
                self.circuit.add_parametric_RY_gate(i, rand_angle())
                self.circuit.add_parametric_RX_gate(i, rand_angle())

    def __call__(self, state: QuantumState):
        self.circuit.update_quantum_state(state)

    def get_n_params(self) -> int:
        return self.circuit.get_parameter_count()

    def set_params(self, theta: List[float]):
        parameter_count = self.get_n_params()

        if len(theta) != parameter_count:
            raise ValueError("number of params must match that of theta", len(
                parameter_count), len(theta))
        for i in range(parameter_count):
            self.circuit.set_parameter(i, theta[i])

    def get_params(self) -> List[float]:
        return [self.circuit.get_parameter(i) for i in range(self.get_n_params())]
