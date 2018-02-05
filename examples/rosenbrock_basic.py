# Py-BOBYQA example: minimize the Rosenbrock function
from __future__ import print_function
import numpy as np
import pybobyqa

# Define the objective function
def rosenbrock(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

# Define the starting point
x0 = np.array([-1.2, 1.0])

# Set random seed (for reproducibility)
np.random.seed(0)

# For optional extra output details
# import logging
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# Call Py-BOBYQA
soln = pybobyqa.solve(rosenbrock, x0)

# Display output
print(soln)

