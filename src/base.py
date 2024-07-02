from enum import Enum, auto

import numpy as np
from numpy import typing as npt
from scipy.optimize import linprog


class Candidate(Enum):
    A = auto()
    B = auto()
    C = auto()


# VOTES = [
#     (2, {Candidate.A: 0, Candidate.B: 1, Candidate.C: 2}),
#     (2, {Candidate.A: 2, Candidate.B: 0, Candidate.C: 1}),
#     (1, {Candidate.A: 1, Candidate.B: 2, Candidate.C: 0}),
# ]
# VOTES = [
#     {Candidate.A: 0, Candidate.B: 1, Candidate.C: 2},
#     {Candidate.A: 1, Candidate.B: 2, Candidate.C: 0},
#     {Candidate.A: 2, Candidate.B: 0, Candidate.C: 1},
# ]
# VOTES = [
#     {Candidate.A: 0, Candidate.B: 1},
#     {Candidate.A: 1, Candidate.B: 0},
# ]
VOTES = [
    (45, {Candidate.A: 0, Candidate.B: 1, Candidate.C: 1}),
    (11, {Candidate.A: 1, Candidate.B: 0, Candidate.C: 1}),
    (15, {Candidate.A: 2, Candidate.B: 0, Candidate.C: 1}),
    (29, {Candidate.A: 2, Candidate.B: 1, Candidate.C: 0}),
]


def pairwise_preferences(
    votes: list[tuple[int, dict[Candidate, int]]],
) -> npt.NDArray[np.int64]:
    n = len(Candidate)
    matrix = np.zeros((n, n), dtype=np.int64)
    for num_voters, vote in votes:
        for i, c1 in enumerate(Candidate):
            for j, c2 in enumerate(Candidate):
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
    A = pairwise_preferences(VOTES)
    print(A)
    print(solve(A))
