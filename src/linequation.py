import numpy as np
from scipy import linalg

# Define the skew-symmetric matrix
A = np.array([[0, -1, 1], [1, 0, -3], [-1, 3, 0]])

# Create the augmented matrix for Av = 0
# We'll add an extra row to enforce the sum-to-1 constraint
B = np.vstack([A, np.ones((1, 3))])

# Create the right-hand side of the equation
# The last element is 1 to enforce the sum-to-1 constraint
b = np.array([0, 0, 0, 1])

# Solve the system of equations
v = linalg.lstsq(B, b)[0]

print("Solution vector:")
print(v)

# Verify that Av = 0 (or very close to 0)
print("\nVerification (Av should be close to 0):")
print(np.dot(A, v))

# Verify that the sum of elements is 1
print("\nSum of vector elements:")
print(np.sum(v))
