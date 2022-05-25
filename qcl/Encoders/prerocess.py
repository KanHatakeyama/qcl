from typing import List
import numpy as np

"""
preprocess funcs
"""


def one_arc_angle(x: List[float], scaling_coeff: float) -> np.array:
    x = np.array(x)*scaling_coeff
    return np.arcsin(x)


def one_unity(x: List[float], scaling_coeff: float) -> np.array:
    x = np.array(x)*scaling_coeff
    return x


def two_arc_angles(x: List[float], scaling_coeff: float) -> np.array:
    x = np.array(x)*scaling_coeff
    angle_y = np.arcsin(x)
    angle_z = np.arccos(x)
    return angle_y, angle_z


def two_unities(x: List[float], scaling_coeff: float) -> np.array:
    x = np.array(x)*scaling_coeff
    x = np.array(x)
    return x, x


def two_x_x2(x: List[float], scaling_coeff: float) -> np.array:
    x = np.array(x)*scaling_coeff
    x = np.array(x)
    return x, x**2


def two_x_inv_x(x: List[float], scaling_coeff: float) -> np.array:
    x = np.array(x)*scaling_coeff
    x = np.array(x)
    return x, x**-1


def two_arc_angle_unity(x: List[float], scaling_coeff: float) -> np.array:
    x = np.array(x)*scaling_coeff
    angle_y = np.arcsin(x)
    return angle_y, x


def three_unity_and_arc_angles(x: List[float], scaling_coeff: float) -> np.array:
    x = np.array(x)*scaling_coeff
    angle_y = np.arcsin(x)
    angle_z = np.arccos(x)
    return x, angle_y, angle_z
