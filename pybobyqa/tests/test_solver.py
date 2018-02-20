"""

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

The development of this software was sponsored by NAG Ltd. (http://www.nag.co.uk)
and the EPSRC Centre For Doctoral Training in Industrially Focused Mathematical
Modelling (EP/L015803/1) at the University of Oxford. Please contact NAG for
alternative licensing.

"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import unittest

import pybobyqa
from pybobyqa.util import sumsq


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


def rosenbrock(x):
    return 100.0 * (x[1]-x[0]**2)**2 + (1-x[0])**2


def rosenbrock_gradient(x):
    return np.array([-2.0* (1.0 - x[0]) - 400*x[0]*(-x[0]**2 + x[1]), 200.0* (-x[0]**2 + x[1])])


def rosenbrock_hessian(x):
    return np.array([[2.0 + 100.0 * (8.0*x[0]**2 - 4.0*(-x[0]**2 + x[1])), -400.0*x[0]], [-400.0*x[0], 200.0]])


class TestRosenbrockGeneric(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 1.0])
        np.random.seed(0)
        soln = pybobyqa.solve(rosenbrock, x0)
        print(soln)
        self.assertTrue(array_compare(soln.x, np.array([1.0, 1.0]), thresh=1e-4), "Wrong xmin")
        self.assertTrue(array_compare(soln.f, rosenbrock(soln.x), thresh=1e-10), "Wrong fmin")
        self.assertTrue(array_compare(soln.gradient, rosenbrock_gradient(soln.x), thresh=1e-2), "Wrong gradient")
        # Hessian entries are quite large, O(100-1000), so can have a fairly large tolerance
        # self.assertTrue(array_compare(soln.hessian, rosenbrock_hessian(soln.x), thresh=1e-0), "Wrong Hessian")
        self.assertLessEqual(np.max(np.abs(rosenbrock_hessian(soln.x) / soln.hessian)) - 1, 1e-1, "Wrong Hessian")
        self.assertTrue(abs(soln.f) < 1e-10, "Wrong fmin")


class TestRosenbrockBounds(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function, where x[1] hits the upper bound
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 0.7])  # standard start point does not satisfy bounds
        lower = np.array([-2.0, -2.0])
        upper = np.array([1.0, 0.9])
        xmin = np.array([0.9486, 0.9])  # approximate
        fmin = rosenbrock(xmin)
        np.random.seed(0)
        soln = pybobyqa.solve(rosenbrock, x0, bounds=(lower, upper))
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(array_compare(soln.f, rosenbrock(soln.x), thresh=1e-10), "Wrong fmin")
        self.assertTrue(array_compare(soln.gradient, rosenbrock_gradient(soln.x), thresh=1e-2), "Wrong gradient")
        # Hessian entries are quite large, O(100-1000), so can have a fairly large tolerance
        # self.assertTrue(array_compare(soln.hessian, rosenbrock_hessian(soln.x), thresh=1e-0), "Wrong Hessian")
        self.assertLessEqual(np.max(np.abs(rosenbrock_hessian(soln.x) / soln.hessian)) - 1, 1e-1, "Wrong Hessian")
        print(soln.hessian)
        print(rosenbrock_hessian(soln.x))
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")


class TestRosenbrockBounds2(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function, where x[0] hits upper bound
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 0.7])  # standard start point too close to upper bounds
        lower = np.array([-2.0, -2.0])
        upper = np.array([0.9, 0.9])
        xmin = np.array([0.9, 0.81])  # approximate
        fmin = rosenbrock(xmin)
        np.random.seed(0)
        soln = pybobyqa.solve(rosenbrock, x0, bounds=(lower, upper))
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")
        self.assertTrue(array_compare(soln.gradient, rosenbrock_gradient(soln.x), thresh=1e-2), "Wrong gradient")
        # Hessian entries are quite large, O(100-1000), so can have a fairly large tolerance (use relative terms)
        self.assertLessEqual(np.max(np.abs(rosenbrock_hessian(soln.x) / soln.hessian)) - 1, 1e-1, "Wrong Hessian")
        # self.assertTrue(array_compare(soln.hessian, rosenbrock_hessian(soln.x), thresh=1e-0), "Wrong Hessian")


class TestLinear(unittest.TestCase):
    # Solve min_x ||Ax-b||^2, for some random A and b
    def runTest(self):
        n, m = 2, 5
        np.random.seed(0)  # (fixing random seed)
        A = np.random.rand(m, n)
        b = np.random.rand(m)
        objfun = lambda x: sumsq(np.dot(A, x) - b)
        gradfun = lambda x: 2.0 * np.dot(A.T, np.dot(A,x)) - 2.0 * np.dot(A.T, b)
        hessfun = 2.0 * np.dot(A.T, A)  # constant Hessian
        xmin = np.linalg.lstsq(A, b)[0]
        fmin = objfun(xmin)
        x0 = np.zeros((n,))
        np.random.seed(0)
        soln = pybobyqa.solve(objfun, x0)
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(array_compare(soln.gradient, gradfun(soln.x), thresh=1e-2), "Wrong gradient")
        self.assertTrue(array_compare(soln.hessian, hessfun, thresh=0.5), "Wrong Hessian")
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")


class TestLinearNp1Model(unittest.TestCase):
    # Solve min_x ||Ax-b||^2, for some random A and b
    def runTest(self):
        n, m = 2, 5
        np.random.seed(0)  # (fixing random seed)
        A = np.random.rand(m, n)
        b = np.random.rand(m)
        objfun = lambda x: sumsq(np.dot(A, x) - b)
        gradfun = lambda x: 2.0 * np.dot(A.T, np.dot(A,x)) - 2.0 * np.dot(A.T, b)
        hessfun = 2.0 * np.dot(A.T, A)  # constant Hessian
        xmin = np.linalg.lstsq(A, b)[0]
        fmin = objfun(xmin)
        x0 = np.zeros((n,))
        np.random.seed(0)
        soln = pybobyqa.solve(objfun, x0, npt=n+1)
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(array_compare(soln.gradient, gradfun(soln.x), thresh=1e-2), "Wrong gradient")
        # self.assertTrue(array_compare(soln.hessian, hessfun, thresh=1e-1), "Wrong Hessian")  # not for linear models
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")


class TestRosenbrockFullyQuadratic(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 1.0])
        np.random.seed(0)
        soln = pybobyqa.solve(rosenbrock, x0, npt=6)
        self.assertTrue(array_compare(soln.x, np.array([1.0, 1.0]), thresh=1e-4), "Wrong xmin")
        self.assertTrue(array_compare(soln.f, rosenbrock(soln.x), thresh=1e-10), "Wrong fmin")
        self.assertTrue(array_compare(soln.gradient, rosenbrock_gradient(soln.x), thresh=1e-2), "Wrong gradient")
        # Hessian entries are quite large, O(100-1000), so can have a fairly large tolerance
        # self.assertTrue(array_compare(soln.hessian, rosenbrock_hessian(soln.x), thresh=1e-0), "Wrong Hessian")
        self.assertLessEqual(np.max(np.abs(rosenbrock_hessian(soln.x) / soln.hessian)) - 1, 1e-1, "Wrong Hessian")
        self.assertTrue(abs(soln.f) < 1e-10, "Wrong fmin")
