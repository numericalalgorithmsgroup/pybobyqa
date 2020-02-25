# Py-BOBYQA example: globally minimize the Freudenstein and Roth function

# Note that Py-BOBYQA only implements a heuristic, so there are no guarantees
# it will find a global minimum. However, by using the seek_global_minimum flag,
# it is more likely to escape local minima if there are better values nearby.

from __future__ import print_function
import numpy as np
import pybobyqa

# Define the objective function
# This function has a local minimum f = 48.98 at x = np.array([11.41, -0.8968])
# and a global minimum f = 0 at x = np.array([5.0, 4.0])
def freudenstein_roth(x):
    r1 = -13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1]
    r2 = -29.0 + x[0] + ((1.0 + x[1]) * x[1] - 14.0) * x[1]
    return r1 ** 2 + r2 ** 2

# Define the starting point
x0 = np.array([5.0, -20.0])

# Define bounds (required for global optimization)
lower = np.array([-30.0, -30.0])
upper = np.array([30.0, 30.0])

print("First run - search for local minimum only")
print("")
soln = pybobyqa.solve(freudenstein_roth, x0, maxfun=500, bounds=(lower, upper))
print(soln)

print("")
print("")

print("Second run - search for global minimum")
print("")
soln = pybobyqa.solve(freudenstein_roth, x0, maxfun=500, bounds=(lower, upper), seek_global_minimum=True)
print(soln)
