import numpy as np


def testing_function(X_array: np.array,
                     mode="sin",
                     ) -> np.array:
    out = np.zeros(X_array.shape[0])

    if mode == "sin1/2":
        for d in range(X_array.shape[1]):
            out += np.sin(X_array[:, d]*np.pi/2)
    elif mode == "sin":
        for d in range(X_array.shape[1]):
            out += np.sin(X_array[:, d]*np.pi)
    elif mode == "sin2":
        for d in range(X_array.shape[1]):
            out += np.sin(X_array[:, d]*np.pi*2)
    elif mode == "linear":
        for d in range(X_array.shape[1]):
            out += X_array[:, d]
    elif mode == "exp":
        for d in range(X_array.shape[1]):
            out += np.exp(X_array[:, d])/np.exp(1)

    elif mode == "linear-sin-interact":
        for d in range(X_array.shape[1]):
            out += X_array[:, d]
            if d < X_array.shape[1]-1:
                out += np.sin(X_array[:, d]*np.pi*2) * \
                    np.sin(X_array[:, d+1]*np.pi*2)

    else:
        raise ValueError(f"unknown mode: {mode}")

    return out
