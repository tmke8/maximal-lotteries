import numpy as np
from numpy import typing as npt
from scipy.optimize import linprog

from src.types import RankedVote, Three, IterableAndSized, Two, Four

# VOTES = [
#     (2, {Three.A: 0, Three.B: 1, Three.C: 2}),
#     (2, {Three.A: 2, Three.B: 0, Three.C: 1}),
#     (1, {Three.A: 1, Three.B: 2, Three.C: 0}),
# ]
# VOTES = [
#     {Three.A: 0, Three.B: 1, Three.C: 2},
#     {Three.A: 1, Three.B: 2, Three.C: 0},
#     {Three.A: 2, Three.B: 0, Three.C: 1},
# ]
# VOTES = [
#     {Two.A: 0, Two.B: 1},
#     {Two.A: 1, Two.B: 0},
# ]
# VOTES = [
#     (45, {Three.A: 0, Three.B: 1, Three.C: 1}),
#     (11, {Three.A: 1, Three.B: 0, Three.C: 1}),
#     (15, {Three.A: 2, Three.B: 0, Three.C: 1}),
#     (29, {Three.A: 2, Three.B: 1, Three.C: 0}),
# ]
# VOTES = [
#     (2, {Four.A: 0, Four.B: 0, Four.C: 3, Four.D: 1}),
#     (1, {Four.A: 0, Four.B: 1, Four.C: 1, Four.D: 1}),
#     (2, {Four.A: 2, Four.B: 2, Four.C: 0, Four.D: 1}),
# ]
VOTES: list[RankedVote[Three]] = [
    (2, {Three.A: 0, Three.B: 1, Three.C: 1}),
    (1, {Three.A: 1, Three.B: 0, Three.C: 0}),
    (2, {Three.A: 1, Three.B: 0, Three.C: 2}),
]


def pairwise_preferences[T](candidates: IterableAndSized[T], votes: list[RankedVote[T]]) -> npt.NDArray[np.int64]:
    n = len(candidates)
    matrix = np.zeros((n, n), dtype=np.int64)
    for num_voters, vote in votes:
        for i, c1 in enumerate(candidates):
            for j, c2 in enumerate(candidates):
                if c1 == c2:
                    continue
                if vote[c2] < vote[c1]:
                    preference = +1
                elif vote[c2] > vote[c1]:
                    preference = -1
                else:
                    preference = 0
                matrix[i, j] += preference * num_voters
    return matrix


def solve(A: npt.NDArray[np.int64]) -> npt.NDArray[np.float64] | None:
    num_vars = A.shape[1]

    # Make matrix positive
    M = A - A.min() + 1

    # Objective function: minimize sum of elements of u
    # linprog will take the dot product of c and u, so a vector of ones results in the sum.
    c = np.ones(num_vars)

    # Inequality constraints: Mu >= 1
    # Note that we negate both sides to fit the linprog format of Ax <= b.
    A_ub = -M
    b_ub = -np.ones(M.shape[0])

    # Non-negativity constraints: u >= 0
    bounds = [(0, None) for _ in range(num_vars)]

    # Solve the linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    print(f"Result: {result.status}")
    print(f"Iterations: {result.nit}")

    if result.success:
        return result.x / result.fun
    else:
        return None


if __name__ == "__main__":
    A = pairwise_preferences(Three, VOTES)
    print(A)
    print(solve(A))
