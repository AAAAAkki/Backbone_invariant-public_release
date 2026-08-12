"""Low-level vector geometry helpers used across the package.

Pure, allocation-light functions for the small set of operations the invariant
computation needs repeatedly: distances, dot/cross products, vector norms,
bond angles, torsion (dihedral) angles, and the law of cosines. They operate
on 1-D NumPy arrays (or array-likes) and return plain floats or arrays.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

#: Type alias for a 1-D array of floating-point coordinates.
FloatArray = npt.NDArray[np.floating[Any]]
#: Type alias for a 1-D array of integers.
IntArray = npt.NDArray[np.integer[Any]]


def get_distance(data_list: list[float]) -> float:
    """Return the Euclidean length (norm) of a vector.

    :param data_list: A vector given as a sequence of coordinates.
    :return: The Euclidean length ``sqrt(sum(x_i**2))``.
    """
    data_list = [i**2 for i in data_list]
    return sum(data_list) ** 0.5


def get_strength(data_list: list[float]):
    """Triangle "strength" ``sigma = (p-a)(p-b)(p-c) / p**2``.

    With ``p = (a+b+c)/2`` the semi-perimeter. A classical triangle-shape
    descriptor equal to zero for degenerate (collinear) triangles.

    :param data_list: Lengths of the three sides of a triangle.
    :return: The strength value.
    """
    p = sum(data_list) * 0.5
    return np.prod([p - i for i in data_list]) / (p**2)


def get_norm_strength(data_list: list[float]) -> np.floating[Any]:
    """Size-normalised triangle strength ``27 * (p-a)(p-b)(p-c) / p**3``.

    Like :func:`get_strength` but normalised to ``[0, 1]`` for an equilateral
    triangle, making it comparable across triangles of different size.

    :param data_list: Lengths of the three sides of a triangle.
    :return: The normalised strength value.
    """
    p = sum(data_list) * 0.5
    return 27 * np.prod([p - i for i in data_list]) / (p**3)


def dot_product(v1: FloatArray, v2: FloatArray) -> float:
    """Dot (scalar) product of two vectors.

    :param v1: 1-D coordinate vector.
    :param v2: 1-D coordinate vector of equal length.
    :return: The dot product ``v1 · v2``.
    """
    return np.dot(v1, v2)


def cross_product(v1: FloatArray, v2: FloatArray) -> FloatArray:
    """Cross (vector) product of two 3-D vectors.

    :param v1: 1-D three-component vector.
    :param v2: 1-D three-component vector.
    :return: The vector ``v1 × v2`` (shape ``(3,)``).
    """
    return np.cross(v1, v2)


def vector_norm(vector: FloatArray) -> np.floating[Any]:
    """Return the Euclidean length of a vector.

    :param vector: A 1-D coordinate vector.
    :return: The norm ``||vector||``.
    """
    return np.linalg.norm(vector)


def vector_round(n: FloatArray, decimal: int = 3) -> FloatArray:
    """Round an array element-wise to a fixed number of decimals.

    :param n: Array (or scalar) to round.
    :param decimal: Number of decimals to keep. Default ``3``.
    :return: *n* rounded to *decimal* places.
    """
    n = np.around(n, decimal)
    return n


def get_angle(v1: FloatArray, v2: FloatArray) -> float:
    """Angle between two vectors, in degrees.

    :param v1: 1-D coordinate vector originating from the shared point.
    :param v2: 1-D coordinate vector originating from the shared point.
    :return: The angle in ``[0, 180]`` degrees, or ``NaN`` if either vector is
        the zero vector.
    """
    v1_norm, v2_norm = vector_norm(v1), vector_norm(v2)
    if not all([v1_norm, v2_norm]):
        return np.nan

    cos = dot_product(v1, v2) / (v1_norm * v2_norm)
    return np.rad2deg(np.arccos(cos))


def get_dihedral_angle(v1: FloatArray, v2: FloatArray, v3: FloatArray) -> float:
    """Dihedral (torsion) angle defined by three consecutive bond vectors.

    For four sequential points ``p1, p2, p3, p4`` the bond vectors are
    ``v1 = p2 - p1``, ``v2 = p3 - p2``, ``v3 = p4 - p3``; this function returns
    the torsion angle of the plane ``(v1, v2)`` against the plane ``(v2, v3)``.

    :param v1: First consecutive 1-D bond vector.
    :param v2: Second consecutive 1-D bond vector.
    :param v3: Third consecutive 1-D bond vector.
    :return: The dihedral angle in degrees, in the range ``(-180, 180]``, or
        ``NaN`` if any vector is the zero vector.
    """
    if not all([vector_norm(v) for v in (v1, v2, v3)]):
        return np.nan

    norm_p1, norm_p2 = cross_product(v1, v2), cross_product(v2, v3)
    p1_x_p2 = cross_product(norm_p1, norm_p2)

    y = dot_product(p1_x_p2, v2) * (1.0 / vector_norm(v2))
    x = dot_product(norm_p1, norm_p2)

    return np.degrees(np.arctan2(y, x))


def hamming_distance(str1: str, str2: str) -> tuple[int, tuple[str, str]]:
    """Hamming distance between two equal-length strings.

    :param str1: First string.
    :param str2: Second string. Must have the same length as *str1*.
    :return: A pair ``(count, diffs)`` where *count* is the number of differing
        positions and *diffs* is ``(str1_diffs, str2_diffs)`` — the mismatched
        characters from each string, concatenated in order.
    :raises ValueError: If *str1* and *str2* have different lengths.
    """
    count = 0
    length = len(str1)
    chain_dif1_list: list[str] = []
    chain_dif2_list: list[str] = []
    if length != len(str2):
        raise ValueError("Input lengths not equal")

    for i in range(length):
        if str1[i] != str2[i]:
            count += 1
            chain_dif1_list.append(str1[i])
            chain_dif2_list.append(str2[i])

    chain_dif1 = "".join(chain_dif1_list)
    chain_dif2 = "".join(chain_dif2_list)
    return count, (chain_dif1, chain_dif2)


def random_matrix(shape: list[int] | tuple, epsilon: float) -> FloatArray:
    """Random perturbation matrix with bounded row norms.

    Each row is a random direction on the unit sphere scaled so that its length
    is uniformly distributed in ``[0, epsilon]`` — i.e. each row lies within a
    ball of radius *epsilon* centred at the origin. Used to generate coordinate
    noise of controlled magnitude.

    :param shape: Output shape; the first entry sets the number of rows (points).
    :param epsilon: Maximum displacement per row.
    :return: An array of the requested *shape* whose rows are bounded by
        *epsilon*.
    """
    u = np.random.normal(0, 1, shape)
    norm = np.sum(u**2, axis=1) ** 0.5
    r = np.random.uniform(0, epsilon, shape[0])
    x = (
        r.reshape(-1, 1) * u / norm.reshape(-1, 1)
    )  # scale vectors to restrict their distance within epsilon
    return x


def get_third_side_length(a: float, b: float, angle_deg: float) -> float:
    """Third side of a triangle via the law of cosines.

    Given two sides *a*, *b* and the angle (in degrees) between them, returns
    the length of the opposite side.

    :param a: Length of the first known side.
    :param b: Length of the second known side.
    :param angle_deg: Angle between *a* and *b*, in degrees.
    :return: The length of the third side.
    """
    angle_rad = np.deg2rad(angle_deg)
    c_squared = a**2 + b**2 - 2 * a * b * np.cos(angle_rad)
    c_squared = max(c_squared, 0.0)
    return np.sqrt(c_squared)


def get_RMSD(x: FloatArray, y: FloatArray, n_dim: int = 3) -> float:
    """Root-mean-square distance between two point sets.

    :param x: First flattened point set.
    :param y: Second flattened point set, of equal size.
    :param n_dim: Dimensionality of each point. Default ``3`` for 3D structures.
    :return: The RMSD ``sqrt(sum((x - y)**2) / N)`` where ``N`` is the number of
        points.
    """
    N = np.size(x) / n_dim  # number of points
    return np.sqrt(np.sum((x - y) ** 2) / N)
