import numpy as np


def concatenate(array):
    lines = []

    for line in array:
        lines.append(np.concatenate(line, axis=1))

    return np.concatenate(lines, axis=0)
