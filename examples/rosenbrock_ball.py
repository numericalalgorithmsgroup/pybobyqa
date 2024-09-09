# Py-BOBYQA example: minimize the Rosenbrock function with bounds and ball constraint
from __future__ import print_function
import numpy as np
import pybobyqa

# Define the objective function
def rosenbrock(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

# Define the starting point
x0 = np.array([-1.2, 1.0])

# Define bound constraints (lower <= x <= upper)
lower = np.array([0.7, -2.0])
upper = np.array([1.0, 2.0])

# Define a Euclidean ball constraint, ||x-center|| <= radius
# This is provided to Py-BOBYQA via a projection operator, which given a point x returns
# the closest point to x which satisfies the constraint
center = np.array([0.5, 1.0])
radius = 0.25
# The projection operator for this constraint is:
ball_proj = lambda x: center + (radius/max(np.linalg.norm(x-center), radius)) * (x-center)

# Many common types of constraints have simple projection operators.
# e.g. lower/upper bounds (but Py-BOBYQA supports this natively, not recommended to use projections)
#      Euclidean balls
#      Linear inequalities, np.dot(a, x) >= b
#      Unit simplex, np.all(x) >= 0 and np.sum(x) <= 1
# For more examples, see
# - https://proximity-operator.net/indicatorfunctions.html
# - Section 6.4.6 (pp. 155-156) of the book: A. Beck, First-Order Methods in Optimization, SIAM (2017)

# For optional extra output details
# import logging
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# Call Py-BOBYQA (with bounds)
soln = pybobyqa.solve(rosenbrock, x0, bounds=(lower, upper), projections=[ball_proj])  # provide list of projection operators

# Display output
print(soln)

