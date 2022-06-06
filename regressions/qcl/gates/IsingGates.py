from ..utils.funcs import rand_angle
from .CNOTRotationGates import CNOTRotationGates
from functools import reduce
from qulacs.gate import X, Z
from qulacs.gate import DenseMatrix
import numpy as np
try:
    import cupy as xp
except:
    import numpy as xp

"""
Ising model type interaction gates

Main ising codes are adopted from works by Qunasys.
https://github.com/qulacs/quantum-native-dojo/blob/master/notebooks/5.2_Quantum_Circuit_Learning.ipynb

"""

I_mat = xp.eye(2, dtype=complex)
X_mat = xp.array(X(0).get_matrix())
Z_mat = xp.array(Z(0).get_matrix())


class IsingGates(CNOTRotationGates):

    def __init__(self,
                 n_qubit: int,
                 depth=2):
        super().__init__(n_qubit, depth)

    def _initialize_gates(self):
        time_evol_gate = prepare_time_evol_gate(self.n_qubit, time_step=0.1)
        for d in range(self.depth):
            # add rotation gates
            self.circuit.add_gate(time_evol_gate)
            for i in range(self.n_qubit):
                # If we observe the qubit of i=0, other qubit rotation in the last depth does not affect the prediction
                if d == self.depth-1 and i > 0:
                    break

                self.circuit.add_parametric_RX_gate(i, rand_angle())
                self.circuit.add_parametric_RZ_gate(i, rand_angle())
                self.circuit.add_parametric_RX_gate(i, rand_angle())


def make_fullgate(list_SiteAndOperator, n_qubit):

    list_Site = [SiteAndOperator[0]
                 for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = []
    cnt = 0
    for i in range(n_qubit):
        if (i in list_Site):
            list_SingleGates.append(list_SiteAndOperator[cnt][1])
            cnt += 1
        else:
            list_SingleGates.append(I_mat)

    return reduce(xp.kron, list_SingleGates)


def prepare_time_evol_gate(n_qubit, time_step):

    ham = xp.zeros((2**n_qubit, 2**n_qubit), dtype=complex)
    for i in range(n_qubit):
        Jx = -1. + 2.*xp.random.rand()
        ham += Jx * make_fullgate([[i, X_mat]], n_qubit)
        for j in range(i+1, n_qubit):
            J_ij = -1. + 2.*xp.random.rand()
            ham += J_ij * \
                make_fullgate([[i, Z_mat], [j, Z_mat]], n_qubit)

    diag, eigen_vecs = xp.linalg.eigh(ham)
    time_evol_op = xp.dot(xp.dot(eigen_vecs, xp.diag(
        xp.exp(-1j*time_step*diag))), eigen_vecs.T.conj())  # e^-iHT

    # in case of cupy
    if type(np) is not type(xp):
        time_evol_op = np.array(xp.asnumpy(time_evol_op))

    time_evol_gate = DenseMatrix(
        [i for i in range(n_qubit)], time_evol_op)

    return time_evol_gate
