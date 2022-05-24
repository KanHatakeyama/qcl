from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
import numpy as np
from ..utils.solver import basinhopping_solver
import copy
from typing import List

"""
Regressor class for Qiskit module

"""


def add_entangle_cnot(circuit: QuantumCircuit, theta_list: List[float], depth=2, n_qubit=2, add_y_gates=False):

    count = 0
    for d in range(depth):

        # add CNOT gates to neighboring bits
        if n_qubit > 1:
            if d % 2 == 0:
                for i in (range(n_qubit-1)):
                    circuit.cx(i, i+1)
                    #circuit.cx(i+1, i)
            else:
                for i in reversed(range(n_qubit-1)):
                    circuit.cx(i+1, i)
                    #circuit.cx(i, i+1)

        # add rotation gates
        for i in range(n_qubit):
            # If we observe the qubit of i=0, other qubit rotation in the last depth does not affect the prediction
            if (d == depth-1) and (i > 0):
                break

            circuit.rx(theta_list[count], i)
            count += 1
            circuit.ry(theta_list[count], i)
            count += 1
            if add_y_gates:
                circuit.rx(theta_list[count], i)
                count += 1

    # print(circuit.draw())


def encode(circuit: QuantumCircuit, x: np.array, n_qubit: int, add_y=False):
    for i in range(n_qubit):
        x_id = i % (x.shape[0])
        #print(x_id, i)
        circuit.rx(x[x_id], i)
        if add_y:
            circuit.ry(x[x_id], i)


class QiskitRegressor:
    def __init__(self, n_qubit: int, x_dim: int,
                 depth: int, add_y_gates=False, solver=None, scale_coeff=2.0, shots=20000, theta_list=None, flip=False):
        self.n_qubit = n_qubit
        self.x_dim = x_dim
        self.depth = depth
        self.scale_coeff = scale_coeff
        self.shots = shots
        self.add_y_gates = add_y_gates
        self.flip = flip
        if theta_list is None:
            self.theta_list = [np.random.random()
                               for _ in range(3*depth*n_qubit)]
        else:
            self.theta_list = copy.copy(theta_list)
        self.simulate = True
        if solver is None:
            self.solver = basinhopping_solver
        else:
            self.solver = solver

    def init_IBMQ(self, machine='ibmq_quito'):
        from qiskit import IBMQ
        self.provider = IBMQ.load_account()  # 3
        self.backend = self.provider.get_backend(machine)  # 4
        print(f"initiated {machine}")

    def predict(self, x_list: List[float]):
        x_array = np.array(x_list)
        x_array = x_array.reshape(-1, self.x_dim)
        return np.array([self.calc_prob(i)*self.scale_coeff for i in x_array])

    def calc_prob(self, x: np.array):

        # prepare cicuit
        c = QuantumCircuit(self.n_qubit, self.n_qubit)
        encode(c, x, self.n_qubit, self.add_y_gates)
        add_entangle_cnot(c, self.theta_list, depth=self.depth,
                          n_qubit=self.n_qubit, add_y_gates=self.add_y_gates)
        c.measure([0], [0])
        #c.measure([0, 1], [0, 1])

        if self.simulate:
            simulator = QasmSimulator()
            compiled_circuit = transpile(c, simulator)
            job = simulator.run(compiled_circuit, shots=self.shots)
        else:
            # use actual quantum machine
            backend = self.backend
            optimized_circuit = transpile(c, backend)  # 5
            job = backend.run(optimized_circuit, shots=self.shots)
            retrieved_job = backend.retrieve_job(job.job_id())
            result = retrieved_job.result()  # 6
            compiled_circuit = optimized_circuit

        result = job.result()
        counts = result.get_counts(compiled_circuit)

        # parse counts
        n_zero_state = counts["0"*self.n_qubit]
        n_one_state = counts["0"*(self.n_qubit-1)+"1"]

        p_zero = n_zero_state/(n_zero_state+n_one_state)
        z = 2*p_zero-1

        if self.flip:
            z = -z

        if not self.simulate:
            print(x, z)

        self.cicuit = c
        return z

    # NOTE: fitting is not easy because prediction is probablistic (i.e., would not converge)
    def fit(self, tr_X: np.array, tr_y: np.array):
        tr_y = tr_y.reshape(-1)

        # define loss func
        def cost_func(theta):
            self.theta_list = theta
            y_pred = self.predict(tr_X)
            loss = ((y_pred - tr_y)**2).mean()
            return loss

        result = self.solver(cost_func, self.theta_list)
        self.theta_list(result.x)
        return self
