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

from pybobyqa.hessian import Hessian


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


class TestBasicInit(unittest.TestCase):
    def runTest(self):
        n = 4
        nvals = n*(n+1)//2
        hess = Hessian(n)
        self.assertEqual(hess.shape(), (nvals,), 'Wrong shape for initialisation')
        self.assertEqual(hess.dim(), n, 'Wrong dimension')
        self.assertEqual(len(hess), nvals, 'Wrong length')
        self.assertTrue(np.all(hess.upper_triangular() == np.zeros((nvals,))), 'Wrong initialised values')


class TestInitFromVector(unittest.TestCase):
    def runTest(self):
        n = 5
        nvals = n*(n+1)//2
        x = np.arange(nvals, dtype=np.float)
        hess = Hessian(n, vals=x)
        self.assertEqual(hess.shape(), (nvals,), 'Wrong shape for initialisation')
        self.assertEqual(hess.dim(), n, 'Wrong dimension')
        self.assertEqual(len(hess), nvals, 'Wrong length')
        self.assertTrue(np.all(hess.upper_triangular() == x), 'Wrong initialised values')


class TestInitFromMatrix(unittest.TestCase):
    def runTest(self):
        n = 3
        nvals = n*(n+1)//2
        A = np.arange(n**2, dtype=np.float).reshape((n,n))
        hess = Hessian(n, vals=A+A.T)  # force symmetric
        self.assertEqual(hess.shape(), (nvals,), 'Wrong shape for initialisation')
        self.assertEqual(hess.dim(), n, 'Wrong dimension')
        self.assertEqual(len(hess), nvals, 'Wrong length')
        self.assertTrue(np.all(hess.upper_triangular() == np.array([0.0, 4.0, 8.0, 8.0, 12.0, 16.0])),
                        'Wrong initialised values')


class TestToFull(unittest.TestCase):
    def runTest(self):
        n = 7
        A = np.arange(n ** 2, dtype=np.float).reshape((n, n))
        H = A + A.T  # force symmetric
        hess = Hessian(n, vals=H)
        self.assertTrue(np.all(hess.as_full() == H), 'Wrong values')


class TestGetElementGood(unittest.TestCase):
    def runTest(self):
        n = 3
        A = np.arange(n ** 2, dtype=np.float).reshape((n, n))
        H = A + A.T  # force symmetric
        hess = Hessian(n, vals=H)
        for i in range(n):
            for j in range(n):
                self.assertEqual(hess.get_element(i, j), H[i,j], 'Wrong value for (i,j)=(%g,%g): got %g, expecting %g'
                                 % (i, j, hess.get_element(i, j), H[i,j]))


class TestGetElementBad(unittest.TestCase):
    def runTest(self):
        n = 4
        A = np.arange(n ** 2, dtype=np.float).reshape((n, n))
        H = A + A.T  # force symmetric
        hess = Hessian(n, vals=H)
        # When testing for assertion errors, need lambda to stop assertion from actually happening
        self.assertRaises(AssertionError, lambda: hess.get_element(-1, 0))
        self.assertRaises(AssertionError, lambda: hess.get_element(-1, 0))
        self.assertRaises(AssertionError, lambda: hess.get_element(-3, n-1))
        self.assertRaises(AssertionError, lambda: hess.get_element(n, 0))
        self.assertRaises(AssertionError, lambda: hess.get_element(n+3, 0))
        self.assertRaises(AssertionError, lambda: hess.get_element(n+7, n-1))
        self.assertRaises(AssertionError, lambda: hess.get_element(0, -1))
        self.assertRaises(AssertionError, lambda: hess.get_element(0, -1))
        self.assertRaises(AssertionError, lambda: hess.get_element(n-1, -3))
        self.assertRaises(AssertionError, lambda: hess.get_element(0, n))
        self.assertRaises(AssertionError, lambda: hess.get_element(0, n+3))
        self.assertRaises(AssertionError, lambda: hess.get_element(n-1, n+7))


class TestSetElementGood(unittest.TestCase):
    def runTest(self):
        n = 3
        A = np.arange(n ** 2, dtype=np.float).reshape((n, n))
        H = A + A.T  # force symmetric
        hess = Hessian(n, vals=H)
        H2 = np.sin(H)
        for i in range(n):
            for j in range(n):
                hess.set_element(i, j, H2[i,j])
        for i in range(n):
            for j in range(n):
                self.assertEqual(hess.get_element(i, j), H2[i, j], 'Wrong value for (i,j)=(%g,%g): got %g, expecting %g'
                                 % (i, j, hess.get_element(i, j), H2[i, j]))


class TestSetElementBad(unittest.TestCase):
    def runTest(self):
        n = 5
        A = np.arange(n ** 2, dtype=np.float).reshape((n, n))
        H = A + A.T  # force symmetric
        hess = Hessian(n, vals=H)
        # When testing for assertion errors, need lambda to stop assertion from actually happening
        self.assertRaises(AssertionError, lambda: hess.set_element(-1, 0, 1.0))
        self.assertRaises(AssertionError, lambda: hess.set_element(-1, 0, 2.0))
        self.assertRaises(AssertionError, lambda: hess.set_element(-3, n - 1, 3.0))
        self.assertRaises(AssertionError, lambda: hess.set_element(n, 0, 4.0))
        self.assertRaises(AssertionError, lambda: hess.set_element(n + 3, 0, -4.0))
        self.assertRaises(AssertionError, lambda: hess.set_element(n + 7, n - 1, 5.0))
        self.assertRaises(AssertionError, lambda: hess.set_element(0, -1, 6.0))
        self.assertRaises(AssertionError, lambda: hess.set_element(0, -1, 7.0))
        self.assertRaises(AssertionError, lambda: hess.set_element(n - 1, -3, -7.0))
        self.assertRaises(AssertionError, lambda: hess.set_element(0, n, -76.3))
        self.assertRaises(AssertionError, lambda: hess.set_element(0, n + 3, 2.8))
        self.assertRaises(AssertionError, lambda: hess.set_element(n - 1, n + 7, -1.0))


class TestMultGood(unittest.TestCase):
    def runTest(self):
        n = 5
        A = np.arange(n ** 2, dtype=np.float).reshape((n, n))
        H = np.sin(A + A.T)  # force symmetric
        hess = Hessian(n, vals=H)
        vec = np.exp(np.arange(n, dtype=np.float))
        hs = np.dot(H, vec)
        self.assertTrue(array_compare(hess*vec, hs, thresh=1e-12), 'Wrong values')


class TestMultBad(unittest.TestCase):
    def runTest(self):
        n = 5
        A = np.arange(n ** 2, dtype=np.float).reshape((n, n))
        H = A + A.T  # force symmetric
        hess = Hessian(n, vals=H)
        # When testing for assertion errors, need lambda to stop assertion from actually happening
        self.assertRaises(AssertionError, lambda: hess * 1.0)
        self.assertRaises(AssertionError, lambda: hess * None)
        self.assertRaises(AssertionError, lambda: hess * [float(i) for i in range(n)])
        self.assertRaises(AssertionError, lambda: hess * np.arange(n-1, dtype=np.float))
        self.assertRaises(AssertionError, lambda: hess * np.arange(n+1, dtype=np.float))


class TestNeg(unittest.TestCase):
    def runTest(self):
        n = 5
        A = np.arange(n ** 2, dtype=np.float).reshape((n, n))
        H = A + A.T  # force symmetric
        hess = Hessian(n, vals=H)
        neghess = -hess
        self.assertTrue(np.allclose(hess.upper_triangular(), -neghess.upper_triangular()), 'Wrong negative values')
