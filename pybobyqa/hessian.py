"""
Hessian Module
====

This module contains a class representing a Hessian matrix, by only storing
the upper triangular elements (since the matrix is symmetric).


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


__all__ = ['Hessian']


class Hessian(object):
    def __init__(self, n, vals=None):
        self.n = n
        if vals is None:
            self.hq = np.zeros((n * (n + 1) // 2,), dtype=np.float)
        else:
            assert isinstance(vals, np.ndarray), "Can only set Hessian from NumPy array"
            assert len(vals.shape) in [1, 2], "Can only set Hessian from vector or matrix"
            if len(vals.shape) == 1:
                assert vals.shape[0] == self.n * (self.n + 1) // 2, "Incompatible n and input vector"
                self.hq = vals
            else:
                # Input was matrix
                assert vals.shape == (self.n, self.n), "Incompatible n and input matrix"
                self.hq = to_upper_triangular_vector(vals)

    def __len__(self):
        return len(self.hq)

    def __neg__(self):  # what is (-hess)?
        return Hessian(self.n, vals=-self.hq)

    def dim(self):
        return self.n

    def shape(self):
        return self.hq.shape

    def upper_triangular(self):
        return self.hq

    def check_valid_index(self, i, j):
        assert i >= 0, "Index i must be >= 0"
        assert i <= self.n-1, "Index i must be <= n-1"
        assert j >= 0, "Index j must be >= 0"
        assert j <= self.n-1, "Index j must be <= n-1"

    def as_full(self):
        A = np.zeros((self.n, self.n))
        ih = -1
        for j in range(self.n):  # j = 0, ..., n-1
            for i in range(j + 1):  # i = 0, ..., j
                ih += 1
                A[i, j] = self.hq[ih]
                A[j, i] = self.hq[ih]
        return A

    def get_element(self, i, j):
        self.check_valid_index(i, j)
        ih = -1
        for k1 in range(self.n):
            for k2 in range(k1 + 1):
                ih += 1
                if (k1 == i and k2 == j) or (k1 == j and k2 == i):
                    return self.hq[ih]
        return None

    def set_element(self, i, j, val):
        self.check_valid_index(i, j)
        ih = -1
        for k1 in range(self.n):
            for k2 in range(k1 + 1):
                ih += 1
                if (k1 == i and k2 == j) or (k1 == j and k2 == i):
                    self.hq[ih] = val
        return

    def vec_mul(self, s):
        # Matrix-vector product
        assert isinstance(s, np.ndarray), "Can only multiply Hessian by a NumPy array"
        assert len(s.shape) == 1, "Can only multiply Hessian by a vector"
        assert s.shape == (self.n,), "Vector has incorrect length (expect %g, got %g)" % (self.n, s.shape[0])
        hs = np.zeros((self.n,))
        ih = -1
        for j in range(self.n):  # j = 0, ..., n-1
            for i in range(j + 1):  # i = 0, ..., j
                ih += 1
                if i < j:
                    hs[j] += self.hq[ih] * s[i]
                hs[i] += self.hq[ih] * s[j]
        return hs

    def __mul__(self, other):
        return self.vec_mul(other)


def to_upper_triangular_vector(A):
    n = A.shape[0]
    hq = np.zeros((n*(n+1)//2,))
    ih = -1
    for j in range(n):  # j = 0, ..., n-1
        for i in range(j + 1):  # i = 0, ..., j
            ih += 1
            hq[ih] = A[i,j]
    return hq
