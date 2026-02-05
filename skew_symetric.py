import numpy as np


def vector_to_skew(vector):
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])


def skew_symetric(vectors):
    """
    Convert multiple 3D vectors to skew-symmetric matrices.

    Args:
        vectors: array-like, shape (3, N)

    Returns:
        Array of skew-symmetric matrices, shape (N, 3, 3)
    """
    if vectors.shape[0] != 3:
        raise ValueError("Input array must have shape (3, N)")

    N = vectors.shape[1]
    result = np.zeros((3, 3, N))

    for i in range(N):
        result[:, :, i] = vector_to_skew(vectors[:, i])

    return result