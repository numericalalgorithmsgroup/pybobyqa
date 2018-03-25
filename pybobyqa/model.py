"""
Model
====

Maintain a class which represents an interpolating set, and its corresponding quadratic model.
This class should calculate the various geometric quantities of interest to us.


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

import logging
from math import sqrt
import numpy as np
import scipy.linalg as LA

from .hessian import Hessian, to_upper_triangular_vector
from .trust_region import trsbox_geometry
from .util import sumsq, model_value

__all__ = ['Model']


class Model(object):
    def __init__(self, npt, x0, f0, xl, xu, f0_nsamples, n=None, abs_tol=-1e20, precondition=True):
        if n is None:
            n = len(x0)
        assert npt >= n + 1, "Require npt >= n+1 for quadratic models"
        assert npt <= (n+1)*(n + 2)//2, "Require npt <= (n+1)(n+2)/2 for quadratic models"
        assert x0.shape == (n,), "x0 has wrong shape (got %s, expect (%g,))" % (str(x0.shape), n)
        assert xl.shape == (n,), "xl has wrong shape (got %s, expect (%g,))" % (str(xl.shape), n)
        assert xu.shape == (n,), "xu has wrong shape (got %s, expect (%g,))" % (str(xu.shape), n)
        self.dim = n
        self.num_pts = npt

        # Initialise to blank some useful stuff
        # Interpolation points
        self.xbase = x0.copy()
        self.sl = xl - self.xbase  # lower bound w.r.t. xbase (require xpt >= sl)
        self.su = xu - self.xbase  # upper bound w.r.t. xbase (require xpt <= su)
        self.points = np.zeros((npt, n))  # interpolation points w.r.t. xbase

        # Function values
        self.f_values = np.inf * np.ones((npt, ))  # overall objective value for each xpt
        self.f_values[0] = f0
        self.kopt = 0  # index of current iterate (should be best value so far)
        self.nsamples = np.zeros((npt,), dtype=np.int)  # number of samples used to evaluate objective at each point
        self.nsamples[0] = f0_nsamples
        self.fbeg = self.f_values[0]  # f(x0), saved to check for sufficient reduction

        # Termination criteria
        self.abs_tol = abs_tol

        # Model information
        self.model_const = 0.0  # constant term for model m(s) = c + J*s
        self.model_grad = np.zeros((n,))  # Jacobian term for model m(s) = c + J*s
        self.model_hess = Hessian(n)

        # Saved point (in absolute coordinates) - always check this value before quitting solver
        self.xsave = None
        self.fsave = None
        self.gradsave = None
        self.hesssave = None
        self.nsamples_save = None

        # Factorisation of interpolation matrix
        self.precondition = precondition  # should the interpolation matrix be preconditioned?
        self.factorisation_current = False
        self.lu = None
        self.piv = None
        self.left_scaling = None
        self.right_scaling = None

    def n(self):
        return self.dim

    def npt(self):
        return self.num_pts

    def xopt(self, abs_coordinates=False):
        return self.xpt(self.kopt, abs_coordinates=abs_coordinates)

    def fopt(self):
        return self.f_values[self.kopt]

    def xpt(self, k, abs_coordinates=False):
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        if not abs_coordinates:
            return self.points[k, :].copy()
        else:
            # Apply bounds and convert back to absolute coordinates
            return self.xbase + np.minimum(np.maximum(self.sl, self.points[k, :]), self.su)

    def fval(self, k):
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        return self.f_values[k]

    def as_absolute_coordinates(self, x):
        # If x were an interpolation point, get the absolute coordinates of x
        return self.xbase + np.minimum(np.maximum(self.sl, x), self.su)

    def xpt_directions(self, include_kopt=True):
        if include_kopt:
            ndirs = self.npt()
        else:
            ndirs = self.npt() - 1

        dirns = np.zeros((ndirs, self.n()))  # vector of directions xpt - xopt, excluding for xopt
        xopt = self.xopt()
        for k in range(self.npt()):
            if not include_kopt and k == self.kopt:
                continue  # skipt
            idx = k if include_kopt or k < self.kopt else k - 1
            dirns[idx, :] = self.xpt(k) - xopt
        return dirns

    def distances_to_xopt(self):
        sq_distances = np.zeros((self.npt(),))
        xopt = self.xopt()
        for k in range(self.npt()):
            sq_distances[k] = sumsq(self.points[k, :] - xopt)
        return sq_distances

    def change_point(self, k, x, f, allow_kopt_update=True):
        # Update point k to x (w.r.t. xbase), with residual values fvec
        assert 0 <= k < self.npt(), "Invalid index %g" % k

        self.points[k, :] = x.copy()
        self.f_values[k] = f
        self.nsamples[k] = 1
        self.factorisation_current = False

        if allow_kopt_update and self.f_values[k] < self.fopt():
            self.kopt = k
        return

    def swap_points(self, k1, k2):
        self.points[[k1, k2], :] = self.points[[k2, k1], :]
        self.f_values[[k1, k2]] = self.f_values[[k2, k1]]
        if self.kopt == k1:
            self.kopt = k2
        elif self.kopt == k2:
            self.kopt = k1
        self.factorisation_current = False
        return

    def add_new_sample(self, k, f_extra):
        # We have resampled at xpt(k) - add this information (f_values is average of all samples)
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        t = float(self.nsamples[k]) / float(self.nsamples[k] + 1)
        self.f_values[k] = t * self.f_values[k] + (1 - t) * f_extra
        self.nsamples[k] += 1

        self.kopt = np.argmin(self.f_values[:self.npt()])  # make sure kopt is always the best value we have
        return

    def add_new_point(self, x, f):
        if self.npt() >= (self.n() + 1) * (self.n() + 2) // 2:
            return False  # cannot add more points

        self.points = np.append(self.points, x.reshape((1, self.n())), axis=0)  # append row to xpt
        self.f_values = np.append(self.f_values, f)  # append entry to f_values
        self.nsamples = np.append(self.nsamples, 1)  # add new sample number
        self.num_pts += 1  # make sure npt is updated

        if f < self.fopt():
            self.kopt = self.npt() - 1

        self.lu_current = False
        return True

    def shift_base(self, xbase_shift):
        # Shifting xbase -> xbase + xbase_shift
        for k in range(self.npt()):
            self.points[k, :] = self.points[k, :] - xbase_shift
        self.xbase += xbase_shift
        self.sl = self.sl - xbase_shift
        self.su = self.su - xbase_shift
        self.factorisation_current = False

        # Update model (always centred on xbase)
        Hx = self.model_hess.vec_mul(xbase_shift)
        self.model_const += np.dot(self.model_grad + 0.5*Hx, xbase_shift)
        self.model_grad += Hx
        return

    def save_point(self, x, f, nsamples, x_in_abs_coords=True):
        if self.fsave is None or f <= self.fsave:
            self.xsave = x.copy() if x_in_abs_coords else self.as_absolute_coordinates(x)
            self.fsave = f
            self.gradsave = self.model_grad.copy()
            self.hesssave = self.model_hess.as_full().copy()
            self.nsamples_save = nsamples
            return True
        else:
            return False  # this value is worse than what we have already - didn't save

    def get_final_results(self):
        # Return x and fval for optimal point (either from xsave+fsave or kopt)
        if self.fsave is None or self.fopt() <= self.fsave:  # optimal has changed since xsave+fsave were last set
            g, hess = self.build_full_model()  # model based at xopt
            return self.xopt(abs_coordinates=True).copy(), self.fopt(), g, hess.as_full(), self.nsamples[self.kopt]
        else:
            return self.xsave, self.fsave, self.gradsave, self.hesssave, self.nsamples_save

    def min_objective_value(self):
        # Get termination criterion for f small: f <= abs_tol
        return self.abs_tol

    def model_value(self, d, d_based_at_xopt=True, with_const_term=False):
        # Model is always centred around xbase
        const = self.model_const if with_const_term else 0.0
        d_to_use = d + self.xopt() if d_based_at_xopt else d
        Hd = self.model_hess.vec_mul(d_to_use)
        return const + np.dot(self.model_grad + 0.5 * Hd, d_to_use)

    def interpolation_matrix(self):
        Y = self.xpt_directions(include_kopt=False).T
        if self.precondition:
            approx_delta = sqrt(np.max(self.distances_to_xopt()))  # largest distance to xopt ~ delta
        else:
            approx_delta = 1.0
        return build_interpolation_matrix(Y, approx_delta=approx_delta)

    def factorise_interp_matrix(self):
        if not self.factorisation_current:
            A, self.left_scaling, self.right_scaling = self.interpolation_matrix()
            self.lu, self.piv = LA.lu_factor(A)
            self.factorisation_current = True
        return

    def solve_system(self, rhs):
        if self.factorisation_current:
            # A(preconditioned) = diag(left_scaling) * A(original) * diag(right_scaling)
            # Solve A(original)\rhs
            return LA.lu_solve((self.lu, self.piv), rhs * self.left_scaling) * self.right_scaling
        else:
            logging.warning("model.solve_system not using factorisation")
            A, left_scaling, right_scaling = self.interpolation_matrix()
            return LA.solve(A, rhs * left_scaling) * right_scaling

    def interpolate_model(self, verbose=False, min_chg_hess=True, get_norm_model_chg=False):
        if verbose:
            A, left_scaling, right_scaling = self.interpolation_matrix()
            interp_cond_num = np.linalg.cond(A)  # scipy.linalg does not have condition number!
        else:
            interp_cond_num = 0.0
        self.factorise_interp_matrix()

        fval_row_idx = np.delete(np.arange(self.npt()), self.kopt)  # indices of all rows except kopt

        if self.npt() == self.n() + 1:
            rhs = self.f_values[fval_row_idx] - self.fopt()
        elif self.npt() == (self.n() + 1)*(self.n() + 2)//2:
            rhs = self.f_values[fval_row_idx] - self.fopt()
        else:
            rhs = np.zeros((self.npt() + self.n() - 1,))
            rhs[:self.npt() - 1] = self.f_values[fval_row_idx] - self.fopt()  # rest of entries are zero
            if min_chg_hess:
                # Modified to be minimum *change* in Hessian, rather than minimum Hessian
                # It's good to see which bits are needed for this specifically (here & 1 line below)
                for t in range(self.npt()-1):
                    dx = self.xpt(fval_row_idx[t]) - self.xopt()
                    rhs[t] = rhs[t] - 0.5 * np.dot(dx, self.model_hess.vec_mul(dx))  # include old Hessian

        try:
            coeffs = self.solve_system(rhs)
        except LA.LinAlgError:
            return False, interp_cond_num, None, None, None  # flag error
        except ValueError:
            return False, interp_cond_num, None, None, None  # flag error (e.g. inf or NaN encountered)

        if not np.all(np.isfinite(coeffs)):  # another check for inf or NaN
            return False, interp_cond_num, None, None, None  # flag error

        # Old gradient and Hessian (save so can compute changes later)
        if verbose or get_norm_model_chg:
            old_model_grad = self.model_grad.copy()
            old_model_hess = self.model_hess.as_full()
        else:
            old_model_grad = None
            old_model_hess = None

        # Build model from coefficients
        self.model_const = self.fopt()  # true in all cases
        if self.npt() == self.n() + 1:
            self.model_grad = coeffs.copy()
            self.model_hess = Hessian(self.n())  # zeros
        elif self.npt() == (self.n() + 1) * (self.n() + 2) // 2:
            self.model_grad = coeffs[:self.n()]
            self.model_hess = Hessian(self.n(), coeffs[self.n():])  # rest of coeffs are upper triangular part of Hess
        else:
            self.model_grad = coeffs[self.npt()-1:]  # last n values
            if min_chg_hess:
                hess_full = self.model_hess.as_full()
            else:
                hess_full = np.zeros((self.n(), self.n()))
            for i in range(self.npt()-1):
                dx = self.xpt(fval_row_idx[i]) - self.xopt()
                hess_full += coeffs[i] * np.outer(dx, dx)
            self.model_hess = Hessian(self.n(), hess_full)

        # Base model at xbase, not xopt (note negative signs)
        xopt = self.xopt()
        Hx = self.model_hess.vec_mul(xopt)
        self.model_const += np.dot(-self.model_grad + 0.5*Hx, xopt)
        self.model_grad += -Hx

        interp_error = 0.0
        norm_chg_grad = 0.0
        norm_chg_hess = 0.0
        if verbose or get_norm_model_chg:
            norm_chg_grad = LA.norm(self.model_grad - old_model_grad)
            norm_chg_hess = LA.norm(self.model_hess.as_full() - old_model_hess, ord='fro')
        if verbose:
            for k in range(self.npt()):
                f_pred = self.model_value(self.xpt(k), d_based_at_xopt=False, with_const_term=True)
                interp_error += self.nsamples[k] * (self.f_values[k] - f_pred)**2
            interp_error = sqrt(interp_error)

        return True, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error  # flag ok

    def build_full_model(self):
        # Make model centred around xopt
        g = self.model_grad + self.model_hess.vec_mul(self.xopt())
        return g, self.model_hess

    def lagrange_polynomial(self, k, factorise_first=True):
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        if factorise_first:
            self.factorise_interp_matrix()

        if k < self.kopt:
            k_row_idx = k
        elif k > self.kopt:
            k_row_idx = k-1
        else:
            k_row_idx = -1  # flag k==kopt

        if self.npt() == self.n() + 1:
            if k_row_idx >= 0:
                rhs = np.zeros((self.n()))
                rhs[k_row_idx] = 1.0
            else:
                rhs = -np.ones((self.n()))
        elif self.npt() == (self.n() + 1) * (self.n() + 2) // 2:
            if k_row_idx >= 0:
                rhs = np.zeros((self.npt()-1))
                rhs[k_row_idx] = 1.0
            else:
                rhs = -np.ones((self.npt()-1))
        else:
            rhs = np.zeros((self.npt() + self.n() - 1,))
            if k_row_idx >= 0:
                rhs[k_row_idx] = 1.0
            else:
                rhs[:self.npt() - 1] = -1.0  # rest of entries are zero

        coeffs = self.solve_system(rhs)

        # Build polynomial from coefficients
        c = 1.0 if k==self.kopt else 0.0  # true in all cases
        if self.npt() == self.n() + 1:
            g = coeffs.copy()
            hess = Hessian(self.n())  # zeros
        elif self.npt() == (self.n() + 1) * (self.n() + 2) // 2:
            g = coeffs[:self.n()]
            hess = Hessian(self.n(), coeffs[self.n():])  # rest of coeffs are upper triangular part of Hess
        else:
            g = coeffs[self.npt() - 1:]  # last n values
            fval_row_idx = np.delete(np.arange(self.npt()), self.kopt)  # indices of all rows except kopt
            hess_full = np.zeros((self.n(), self.n()))
            for i in range(self.npt() - 1):
                dx = self.xpt(fval_row_idx[i]) - self.xopt()
                hess_full += coeffs[i] * np.outer(dx, dx)
            hess = Hessian(self.n(), hess_full)

        # (c, g, hess) currently based around xopt
        return c, g, hess

    def poisedness_constant(self, delta, xbase=None, xbase_in_abs_coords=True):
        # Calculate the poisedness constant of the current interpolation set in B(xbase, delta)
        # if xbase is None, use self.xopt()
        overall_max = None
        if xbase is None:
            xbase = self.xopt()
        elif xbase_in_abs_coords:
            xbase = xbase - self.xbase  # shift to correct position
        for k in range(self.npt()):
            c, g, hess = self.lagrange_polynomial(k, factorise_first=True)  # based at self.xopt()
            # Switch base of poly from xopt to xbase, as required by trsbox_geometry
            base_chg = self.xopt() - xbase
            Hx = hess.vec_mul(base_chg)
            c += np.dot(-g + 0.5 * Hx, base_chg)
            g += -Hx
            xmax = trsbox_geometry(xbase, c, g, hess, self.sl, self.su, delta)
            lmax = abs(c + model_value(g, hess, xmax-xbase))  # evaluate Lagrange poly
            if overall_max is None or lmax > overall_max:
                overall_max = lmax
        return overall_max


def build_interpolation_matrix(Y, approx_delta=1.0):
    # Y has columns Y[:,t] = yt - xk
    n, p = Y.shape  # p = npt-1
    assert n + 1 <= p + 1 <= (n + 1) * (n + 2) // 2, "npt must be in range [n+1, (n+1)(n+2)/2]"

    # What scaling was applied to each part of the matrix?
    # A(scaled) = diag(left_scaling) * A(unscaled) * diag(right_scaling)

    if p == n:  # linear models
        A = Y.T / approx_delta
        left_scaling = np.ones((n,))  # no left scaling
        right_scaling = np.ones((n,)) / approx_delta
    elif p + 1 == (n+1)*(n+2)//2:  # fully quadratic models
        A = np.zeros((p, p))
        A[:,:n] = Y.T / approx_delta
        for i in range(p):
            A[i, n:] = to_upper_triangular_vector(np.outer(Y[:,i], Y[:,i]) - 0.5*np.diag(np.square(Y[:,i]))) / (approx_delta**2)
        left_scaling = np.ones((p,))  # no left scaling
        right_scaling = np.ones((p,))
        right_scaling[:n] = 1.0 / approx_delta
        right_scaling[n:] = 1.0 / (approx_delta**2)
    else:  # underdetermined quadratic models
        A = np.zeros((p + n, p + n))
        for i in range(p):
            for j in range(p):
                A[i,j] = 0.5*np.dot(Y[:,i], Y[:,j])**2 / (approx_delta**4)
        A[:p,p:] = Y.T / approx_delta
        A[p:,:p] = Y / approx_delta
        left_scaling = np.ones((p+n,))
        right_scaling = np.ones((p + n,))
        left_scaling[:p] = 1.0 / (approx_delta**2)
        left_scaling[p:] = approx_delta
        right_scaling[:p] = 1.0 / (approx_delta**2)
        right_scaling[p:] = approx_delta
    return A, left_scaling, right_scaling

