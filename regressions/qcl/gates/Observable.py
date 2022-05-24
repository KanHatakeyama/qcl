
from qulacs import Observable

"""
Z-observable
"""


class ZObservable:
    def __init__(self, n_qubit: int, coeff=2.0, command="Z 0"):
        self.observable = Observable(n_qubit)
        self.observable.add_operator(coeff, command)

    def __call__(self, state) -> float:
        return self.observable.get_expectation_value(state)
