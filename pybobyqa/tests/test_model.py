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

from math import sqrt, sin
import numpy as np
import unittest

from pybobyqa.hessian import Hessian
from pybobyqa.model import Model
from pybobyqa.util import sumsq, model_value


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


def rosenbrock_residuals(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def rosenbrock(x):
    return sumsq(rosenbrock_residuals(x))


def objfun(x):
    # An arbitrary-dimension objective function
    return sin(np.dot(x, np.arange(1,len(x)+1,dtype=np.float)))  # f(x1,...,xn) = sin(x1 + 2*x2 + ... + n*xn)


class TestAddValues(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e20 * np.ones((n,))
        xu = 1e20 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1)
        self.assertEqual(model.npt(), npt, 'Wrong npt after initialisation')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x0), 'Wrong xopt after initialisation')
        self.assertTrue(array_compare(model.fopt(), rosenbrock(x0)), 'Wrong fopt after initialisation')
        # Now add better point
        x1 = np.array([1.0, 0.9])
        rvec = rosenbrock(x1)
        model.change_point(1, x1 - model.xbase, rvec, allow_kopt_update=True)
        self.assertEqual(model.npt(), npt, 'Wrong npt after x1')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after x1')
        self.assertTrue(array_compare(model.fopt(), rosenbrock(x1)), 'Wrong fopt after x1')
        # Now add worse point
        x2 = np.array([2.0, 0.9])
        rvec = rosenbrock(x2)
        model.change_point(2, x2 - model.xbase, rvec, allow_kopt_update=True)
        self.assertEqual(model.npt(), npt, 'Wrong npt after x2')
        self.assertTrue(array_compare(model.xpt(0, abs_coordinates=True), x0), 'Wrong xpt(0) after x2')
        self.assertTrue(array_compare(model.xpt(1, abs_coordinates=True), x1), 'Wrong xpt(1) after x2')
        self.assertTrue(array_compare(model.xpt(2, abs_coordinates=True), x2), 'Wrong xpt(2) after x2')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after x2')
        self.assertTrue(array_compare(model.fopt(), rosenbrock(x1)), 'Wrong fopt after x2')
        # Now add best point (but don't update kopt)
        x3 = np.array([1.0, 1.0])
        rvec = rosenbrock(x3)
        model.change_point(0, x3 - model.xbase, rvec, allow_kopt_update=False)  # full: overwrite x0
        self.assertEqual(model.npt(), npt, 'Wrong npt after x3')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after x3')
        self.assertTrue(array_compare(model.fopt(), rosenbrock(x1)), 'Wrong fopt after x3')
        self.assertAlmostEqual(model.fopt(), rosenbrock(x1), msg='Wrong fopt after x3')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), model.as_absolute_coordinates(model.xopt())),
                        'Comparison wrong after x3')
        dirns = model.xpt_directions(include_kopt=True)
        self.assertTrue(array_compare(x3 - x1, dirns[0, :]), 'Wrong dirn 0')
        self.assertTrue(array_compare(x1 - x1, dirns[1, :]), 'Wrong dirn 1')
        self.assertTrue(array_compare(x2 - x1, dirns[2, :]), 'Wrong dirn 2')
        dirns = model.xpt_directions(include_kopt=False)
        self.assertTrue(array_compare(x3 - x1, dirns[0, :]), 'Wrong dirn 0 (no kopt)')
        # self.assertTrue(array_compare(x1 - x1, dirns[1, :]), 'Wrong dirn 1')
        self.assertTrue(array_compare(x2 - x1, dirns[1, :]), 'Wrong dirn 1 (no kopt)')


class TestSwap(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e20 * np.ones((n,))
        xu = 1e20 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1)
        # Now add better point
        x1 = np.array([1.0, 0.9])
        f1 = rosenbrock(x1)
        model.change_point(1, x1 - model.xbase, f1, allow_kopt_update=True)
        # Now add worse point
        x2 = np.array([2.0, 0.9])
        f2 = rosenbrock(x2)
        model.change_point(2, x2 - model.xbase, f2, allow_kopt_update=True)
        model.swap_points(0, 2)
        self.assertTrue(array_compare(model.xpt(0, abs_coordinates=True), x2), 'Wrong xpt(0) after swap 1')
        self.assertTrue(array_compare(model.xpt(1, abs_coordinates=True), x1), 'Wrong xpt(1) after swap 1')
        self.assertTrue(array_compare(model.xpt(2, abs_coordinates=True), x0), 'Wrong xpt(2) after swap 1')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after swap 1')
        model.swap_points(1, 2)
        self.assertTrue(array_compare(model.xpt(0, abs_coordinates=True), x2), 'Wrong xpt(0) after swap 2')
        self.assertTrue(array_compare(model.xpt(1, abs_coordinates=True), x0), 'Wrong xpt(1) after swap 2')
        self.assertTrue(array_compare(model.xpt(2, abs_coordinates=True), x1), 'Wrong xpt(2) after swap 2')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after swap 2')


class TestBasicManipulation(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1)
        self.assertTrue(array_compare(model.sl, xl - x0), 'Wrong sl after initialisation')
        self.assertTrue(array_compare(model.su, xu - x0), 'Wrong su after initialisation')
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        self.assertTrue(array_compare(model.as_absolute_coordinates(x1 - x0), x1), 'Wrong abs coords')
        self.assertTrue(array_compare(model.as_absolute_coordinates(np.array([-1e3, 1e3])-x0), np.array([-1e2, 1e2])),
                        'Bad abs coords with bounds')
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        sqdists = model.distances_to_xopt()
        self.assertAlmostEqual(sqdists[0], sumsq(x0 - x1), msg='Wrong distance 0')
        self.assertAlmostEqual(sqdists[1], sumsq(x1 - x1), msg='Wrong distance 1')
        self.assertAlmostEqual(sqdists[2], sumsq(x2 - x1), msg='Wrong distance 2')
        model.add_new_sample(0, rosenbrock(x0))
        self.assertEqual(model.nsamples[0], 2, 'Wrong number of samples 0')
        self.assertEqual(model.nsamples[1], 1, 'Wrong number of samples 1')
        self.assertEqual(model.nsamples[2], 1, 'Wrong number of samples 2')
        for i in range(50):
            model.add_new_sample(0, 0.0)
        self.assertEqual(model.kopt, 0, 'Wrong kopt after bad resampling')
        self.assertTrue(array_compare(model.fopt(), 2*rosenbrock(x0)/52), 'Wrong fopt after bad resampling')
        d = np.array([10.0, 10.0])
        dirns_old = model.xpt_directions(include_kopt=True)
        model.shift_base(d)
        dirns_new = model.xpt_directions(include_kopt=True)
        self.assertTrue(array_compare(model.xbase, x0 + d), 'Wrong new base')
        self.assertEqual(model.kopt, 0, 'Wrong kopt after shift base')
        for i in range(3):
            self.assertTrue(array_compare(dirns_old[i, :], dirns_new[i, :]), 'Wrong dirn %i after shift base' % i)
        self.assertTrue(array_compare(model.sl, xl - x0 - d), 'Wrong sl after shift base')
        self.assertTrue(array_compare(model.su, xu - x0 - d), 'Wrong su after shift base')
        # save_point and get_final_results
        model.change_point(0, x0 - model.xbase, rosenbrock(x0))  # revert after resampling
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))  # revert after resampling
        x, f, gradmin, hessmin, nsamples = model.get_final_results()
        self.assertTrue(array_compare(x, x1), 'Wrong final x')
        self.assertAlmostEqual(rosenbrock(x1), f, msg='Wrong final f')
        self.assertTrue(array_compare(np.zeros((2,)), gradmin), 'Wrong final gradmin')
        self.assertTrue(array_compare(np.zeros((2,2)), hessmin), 'Wrong final hessmin')
        self.assertEqual(1, nsamples, 'Wrong final nsamples')
        self.assertIsNone(model.xsave, 'xsave not none after initialisation')
        self.assertIsNone(model.fsave, 'fsave not none after initialisation')
        self.assertIsNone(model.nsamples_save, 'nsamples_save not none after initialisation')
        model.save_point(x0, rosenbrock(x0), 1, x_in_abs_coords=True)
        self.assertTrue(array_compare(model.xsave, x0), 'Wrong xsave after saving')
        self.assertAlmostEqual(model.fsave, rosenbrock(x0), msg='Wrong fsave after saving')
        self.assertEqual(model.nsamples_save, 1, 'Wrong nsamples_save after saving')
        x, f, gradmin, hessmin, nsamples = model.get_final_results()
        self.assertTrue(array_compare(x, x1), 'Wrong final x after saving')
        self.assertAlmostEqual(rosenbrock(x1), f, msg='Wrong final f after saving')
        self.assertEqual(1, nsamples, 'Wrong final nsamples after saving')
        model.save_point(x2 - model.xbase, 0.0, 2, x_in_abs_coords=False)
        self.assertTrue(array_compare(model.xsave, x2), 'Wrong xsave after saving 2')
        self.assertAlmostEqual(model.fsave, 0.0, msg='Wrong fsave after saving 2')
        self.assertEqual(model.nsamples_save, 2, 'Wrong nsamples_save after saving 2')
        x, f, gradmin, hessmin, nsamples = model.get_final_results()
        self.assertTrue(array_compare(x, x2), 'Wrong final x after saving 2')
        self.assertAlmostEqual(f, 0.0, msg='Wrong final f after saving 2')
        self.assertEqual(2, nsamples, 'Wrong final nsamples after saving 2')
        model.save_point(x0, rosenbrock(x0), 3, x_in_abs_coords=True)  # try to re-save a worse value
        self.assertTrue(array_compare(model.xsave, x2), 'Wrong xsave after saving 3')
        self.assertAlmostEqual(model.fsave, 0.0, msg='Wrong fsave after saving 3')
        self.assertEqual(model.nsamples_save, 2, 'Wrong nsamples_save after saving 3')


class TestAveraging(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1)
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([1.0, 1.0])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        self.assertEqual(model.kopt, 2, 'Wrong kopt before resampling')
        # Originally, x2 is the ideal point
        # Here, testing that kopt moves back to x1 after adding heaps of bad x2 samples
        for i in range(10):
            model.add_new_sample(2, 5.0)
        self.assertEqual(model.kopt, 1, 'Wrong kopt after resampling')


class TestMinObjValue(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1)
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        self.assertAlmostEqual(model.min_objective_value(), -1e20, msg='Wrong min obj value')
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1, abs_tol=1.0)
        self.assertAlmostEqual(model.min_objective_value(), 1.0, msg='Wrong min obj value 3')


class TestInterpMatrixLinear(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1, precondition=False)
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        A, left_scaling, right_scaling = model.interpolation_matrix()
        A_expect = np.zeros((2, 2))
        A_expect[0, :] = x0 - x1  # x1 is xopt in this situation
        A_expect[1, :] = x2 - x1
        self.assertTrue(array_compare(A, A_expect), 'Interp matrix 1')
        # For reference: model based around model.xbase
        interp_ok, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error = model.interpolate_model()
        self.assertTrue(interp_ok, 'Interpolation failed')
        self.assertAlmostEqual(interp_error, 0.0, msg='Expect exact interpolation')
        self.assertAlmostEqual(model.model_const, rosenbrock(model.xbase), msg='Wrong constant term')
        self.assertTrue(array_compare(model.model_value(x1 - model.xbase, d_based_at_xopt=False, with_const_term=True),
                                      rosenbrock(x1), thresh=1e-10), 'Wrong x1')  # allow some inexactness
        self.assertTrue(array_compare(model.model_value(x2 - model.xbase, d_based_at_xopt=False, with_const_term=True),
                                      rosenbrock(x2), thresh=1e-10), 'Wrong x2')
        # Test some other parameter settings for model.model_value()
        self.assertTrue(array_compare(model.model_value(x2 - x1, d_based_at_xopt=True, with_const_term=True),
                                      rosenbrock(x2), thresh=1e-10), 'Wrong x2 (from xopt)')
        self.assertTrue(array_compare(model.model_value(x2 - x1, d_based_at_xopt=True, with_const_term=False),
                                      rosenbrock(x2)-rosenbrock(model.xbase), thresh=1e-10), 'Wrong x2 (no constant)')
        self.assertTrue(array_compare(model.model_value(x2 - model.xbase, d_based_at_xopt=False, with_const_term=False),
                                rosenbrock(x2) - rosenbrock(model.xbase), thresh=1e-10), 'Wrong x2 (no constant v2)')
        g, hess = model.build_full_model()
        self.assertTrue(np.allclose(g, model.model_grad + model.model_hess.vec_mul(model.xopt(abs_coordinates=False))),
                        'Bad gradient')
        self.assertTrue(np.allclose(hess.as_full(), model.model_hess.as_full()), 'Bad Hessian')


class TestInterpMatrixUnderdeterminedQuadratic(unittest.TestCase):
    def runTest(self):
        n = 2
        npt = n+2
        x0 = np.array([1.0, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, objfun(x0), xl, xu, 1, precondition=False)
        x1 = x0 + np.array([1.0, 0.0])
        model.change_point(1, x1 - model.xbase, objfun(x1))
        x2 = x0 + np.array([0.1, 0.9])
        model.change_point(2, x2 - model.xbase, objfun(x2))
        x3 = x0 + np.array([-0.1, 0.0])
        model.change_point(3, x3 - model.xbase, objfun(x3))

        # x2 is xopt in this situation
        self.assertTrue(model.kopt == 2, 'Wrong xopt')
        xs = [x0, x1, x3]
        xopt = x2
        nxs = len(xs)
        A = np.zeros((nxs+n,nxs+n))
        for i in range(nxs):
            for j in range(nxs):
                A[i,j] = 0.5 * np.dot(xs[i]-xopt, xs[j]-xopt)**2
            A[i,nxs:] = xs[i] - xopt
            A[nxs:,i] = xs[i] - xopt

        A2, left_scaling, right_scaling = model.interpolation_matrix()
        # print("Expect", A)
        # print("Got", A2)
        self.assertTrue(np.allclose(A, A2), 'Interp matrix 1')

        # For reference: model based around model.xbase
        interp_ok, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error = model.interpolate_model(verbose=True)
        self.assertTrue(interp_ok, 'Interpolation failed')
        self.assertAlmostEqual(interp_error, 0.0, msg='Expect exact interpolation')
        self.assertAlmostEqual(norm_chg_grad, np.linalg.norm(model.model_grad))
        self.assertAlmostEqual(norm_chg_hess, np.linalg.norm(model.model_hess.as_full(), ord='fro'))
        self.assertAlmostEqual(model.model_const, objfun(model.xbase), msg='Wrong constant term')
        for xi in [x0, x1, x2, x3]:
            self.assertAlmostEqual(model.model_value(xi - model.xbase, d_based_at_xopt=False, with_const_term=True),
                                      objfun(xi), msg='Wrong interp value at %s' % str(xi))
        # Test some other parameter settings for model.model_value()
        print("Ignore after here")
        g, hess = model.build_full_model()
        self.assertTrue(np.allclose(g, model.model_grad + model.model_hess.vec_mul(model.xopt(abs_coordinates=False))),
                        'Bad gradient')
        self.assertTrue(np.allclose(hess.as_full(), model.model_hess.as_full()), 'Bad Hessian')

        # Build a new model
        model2 = Model(npt, x0, objfun(x0), xl, xu, 1, precondition=False)
        model2.change_point(1, x1 - model.xbase, objfun(x1))
        model2.change_point(2, x2 - model.xbase, objfun(x2))
        model2.change_point(3, x3 - model.xbase, objfun(x3))
        # Force Hessian to be something else
        model2.model_hess = Hessian(n, vals=np.eye(n))
        A2, left_scaling, right_scaling = model2.interpolation_matrix()
        self.assertTrue(np.allclose(A, A2), 'Interp matrix 2')
        interp_ok, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error = model2.interpolate_model()
        self.assertTrue(interp_ok, 'Interpolation failed')
        self.assertAlmostEqual(interp_error, 0.0, msg='Expect exact interpolation')
        self.assertAlmostEqual(model2.model_const, objfun(model2.xbase), msg='Wrong constant term')
        for xi in [x0, x1, x2, x3]:
            self.assertAlmostEqual(model2.model_value(xi - model2.xbase, d_based_at_xopt=False, with_const_term=True),
                                   objfun(xi), msg='Wrong interp value at %s' % str(xi))

        # Compare distance of hessians
        h1 = Hessian(n).as_full()
        h2 = Hessian(n, vals=np.eye(n)).as_full()
        self.assertLessEqual(np.linalg.norm(model.model_hess.as_full()-h1, ord='fro'),
                             np.linalg.norm(model2.model_hess.as_full()-h1, ord='fro'), 'Not min frob Hess 1')
        self.assertLessEqual(np.linalg.norm(model2.model_hess.as_full() - h2, ord='fro'),
                             np.linalg.norm(model.model_hess.as_full() - h2, ord='fro'), 'Not min frob Hess 2')
        # print(model.model_hess.as_full())
        # print(model2.model_hess.as_full())

        # Build a new model
        model3 = Model(npt, x0, objfun(x0), xl, xu, 1, precondition=False)
        model3.change_point(1, x1 - model.xbase, objfun(x1))
        model3.change_point(2, x2 - model.xbase, objfun(x2))
        model3.change_point(3, x3 - model.xbase, objfun(x3))
        # Force Hessian to be something else
        model3.model_hess = Hessian(n, vals=np.eye(n))
        A2, left_scaling, right_scaling = model3.interpolation_matrix()
        self.assertTrue(np.allclose(A, A2), 'Interp matrix 3')
        interp_ok, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error = model3.interpolate_model(min_chg_hess=False)
        self.assertTrue(interp_ok, 'Interpolation failed')
        self.assertAlmostEqual(interp_error, 0.0, msg='Expect exact interpolation')
        self.assertAlmostEqual(model3.model_const, objfun(model3.xbase), msg='Wrong constant term')
        for xi in [x0, x1, x2, x3]:
            self.assertAlmostEqual(model3.model_value(xi - model3.xbase, d_based_at_xopt=False, with_const_term=True),
                                   objfun(xi), msg='Wrong interp value at %s' % str(xi))
        self.assertTrue(np.allclose(model.model_hess.as_full(), model3.model_hess.as_full()),
                        'min_chg_hess=False not working')


class TestInterpMatrixUnderdeterminedQuadratic2(unittest.TestCase):
    def runTest(self):
        n = 2
        npt = 2*n+1
        x0 = np.array([1.0, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, objfun(x0), xl, xu, 1, precondition=False)
        x1 = x0 + np.array([1.0, 0.0])
        model.change_point(1, x1 - model.xbase, objfun(x1))
        x2 = x0 + np.array([0.1, 0.9])
        model.change_point(2, x2 - model.xbase, objfun(x2))
        x3 = x0 + np.array([-0.1, 0.0])
        model.change_point(3, x3 - model.xbase, objfun(x3))
        x4 = x0 + np.array([-0.1, 2.0])
        model.change_point(4, x4 - model.xbase, objfun(x4))

        # x2 is xopt in this situation
        self.assertTrue(model.kopt == 2, 'Wrong xopt')
        xs = [x0, x1, x3, x4]
        xopt = x2
        nxs = len(xs)
        A = np.zeros((nxs+n,nxs+n))
        for i in range(nxs):
            for j in range(nxs):
                A[i,j] = 0.5 * np.dot(xs[i]-xopt, xs[j]-xopt)**2
            A[i,nxs:] = xs[i] - xopt
            A[nxs:,i] = xs[i] - xopt

        A2, left_scaling, right_scaling = model.interpolation_matrix()
        # print("Expect", A)
        # print("Got", A2)
        self.assertTrue(np.allclose(A, A2), 'Interp matrix 1')

        # For reference: model based around model.xbase
        interp_ok, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error = model.interpolate_model(verbose=True)
        self.assertTrue(interp_ok, 'Interpolation failed')
        self.assertAlmostEqual(interp_error, 0.0, msg='Expect exact interpolation')
        self.assertAlmostEqual(norm_chg_grad, np.linalg.norm(model.model_grad))
        self.assertAlmostEqual(norm_chg_hess, np.linalg.norm(model.model_hess.as_full(), ord='fro'))
        self.assertAlmostEqual(model.model_const, objfun(model.xbase), msg='Wrong constant term')
        for xi in [x0, x1, x2, x3, x4]:
            self.assertAlmostEqual(model.model_value(xi - model.xbase, d_based_at_xopt=False, with_const_term=True),
                                      objfun(xi), msg='Wrong interp value at %s' % str(xi))
        # Test some other parameter settings for model.model_value()
        g, hess = model.build_full_model()
        self.assertTrue(np.allclose(g, model.model_grad + model.model_hess.vec_mul(model.xopt(abs_coordinates=False))),
                        'Bad gradient')
        self.assertTrue(np.allclose(hess.as_full(), model.model_hess.as_full()), 'Bad Hessian')

        # Build a new model
        model2 = Model(npt, x0, objfun(x0), xl, xu, 1, precondition=False)
        model2.change_point(1, x1 - model.xbase, objfun(x1))
        model2.change_point(2, x2 - model.xbase, objfun(x2))
        model2.change_point(3, x3 - model.xbase, objfun(x3))
        model2.change_point(4, x4 - model.xbase, objfun(x4))
        # Force Hessian to be something else
        model2.model_hess = Hessian(n, vals=np.eye(n))
        A2, left_scaling, right_scaling = model2.interpolation_matrix()
        self.assertTrue(np.allclose(A, A2), 'Interp matrix 2')
        interp_ok, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error = model2.interpolate_model()
        self.assertTrue(interp_ok, 'Interpolation failed')
        self.assertAlmostEqual(interp_error, 0.0, msg='Expect exact interpolation')
        self.assertAlmostEqual(model2.model_const, objfun(model2.xbase), msg='Wrong constant term')
        for xi in [x0, x1, x2, x3, x4]:
            self.assertAlmostEqual(model2.model_value(xi - model2.xbase, d_based_at_xopt=False, with_const_term=True),
                                   objfun(xi), msg='Wrong interp value at %s' % str(xi))

        # Compare distance of hessians
        h1 = Hessian(n).as_full()
        h2 = Hessian(n, vals=np.eye(n)).as_full()
        self.assertLessEqual(np.linalg.norm(model.model_hess.as_full()-h1, ord='fro'),
                             np.linalg.norm(model2.model_hess.as_full()-h1, ord='fro'), 'Not min frob Hess 1')
        self.assertLessEqual(np.linalg.norm(model2.model_hess.as_full() - h2, ord='fro'),
                             np.linalg.norm(model.model_hess.as_full() - h2, ord='fro'), 'Not min frob Hess 2')
        # print(model.model_hess.as_full())
        # print(model2.model_hess.as_full())

        # Build a new model
        model3 = Model(npt, x0, objfun(x0), xl, xu, 1, precondition=False)
        model3.change_point(1, x1 - model.xbase, objfun(x1))
        model3.change_point(2, x2 - model.xbase, objfun(x2))
        model3.change_point(3, x3 - model.xbase, objfun(x3))
        model3.change_point(4, x4 - model.xbase, objfun(x4))
        # Force Hessian to be something else
        model3.model_hess = Hessian(n, vals=np.eye(n))
        A2, left_scaling, right_scaling = model3.interpolation_matrix()
        self.assertTrue(np.allclose(A, A2), 'Interp matrix 3')
        interp_ok, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error = model3.interpolate_model(min_chg_hess=False)
        self.assertTrue(interp_ok, 'Interpolation failed')
        self.assertAlmostEqual(interp_error, 0.0, msg='Expect exact interpolation')
        self.assertAlmostEqual(model3.model_const, objfun(model3.xbase), msg='Wrong constant term')
        for xi in [x0, x1, x2, x3, x4]:
            self.assertAlmostEqual(model3.model_value(xi - model3.xbase, d_based_at_xopt=False, with_const_term=True),
                                   objfun(xi), msg='Wrong interp value at %s' % str(xi))
        self.assertTrue(np.allclose(model.model_hess.as_full(), model3.model_hess.as_full()),
                        'min_chg_hess=False not working')


class TestInterpMatrixFullQuadratic(unittest.TestCase):
    def runTest(self):
        n = 2
        npt = (n+1) * (n+2) // 2
        x0 = np.array([1.0, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, objfun(x0), xl, xu, 1)
        x1 = x0 + np.array([1.0, 0.0])
        model.change_point(1, x1 - model.xbase, objfun(x1))
        x2 = x0 + np.array([0.1, 0.9])
        model.change_point(2, x2 - model.xbase, objfun(x2))
        x3 = x0 + np.array([-0.1, 0.0])
        model.change_point(3, x3 - model.xbase, objfun(x3))
        x4 = x0 + np.array([-0.1, 2.0])
        model.change_point(4, x4 - model.xbase, objfun(x4))
        x5 = x0 + np.array([-1.1, 1.0])
        model.change_point(5, x5 - model.xbase, objfun(x5))

        # For reference: model based around model.xbase
        interp_ok, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error = model.interpolate_model(verbose=True)
        self.assertTrue(interp_ok, 'Interpolation failed')
        self.assertAlmostEqual(interp_error, 0.0, msg='Expect exact interpolation')
        self.assertAlmostEqual(norm_chg_grad, np.linalg.norm(model.model_grad))
        self.assertAlmostEqual(norm_chg_hess, np.linalg.norm(model.model_hess.as_full(), ord='fro'))
        self.assertAlmostEqual(model.model_const, objfun(model.xbase), msg='Wrong constant term')
        for xi in [x0, x1, x2, x3, x4, x5]:
            self.assertAlmostEqual(model.model_value(xi - model.xbase, d_based_at_xopt=False, with_const_term=True),
                                      objfun(xi), msg='Wrong interp value at %s' % str(xi))
        # Test some other parameter settings for model.model_value()
        g, hess = model.build_full_model()
        self.assertTrue(np.allclose(g, model.model_grad + model.model_hess.vec_mul(model.xopt(abs_coordinates=False))),
                        'Bad gradient')
        self.assertTrue(np.allclose(hess.as_full(), model.model_hess.as_full()), 'Bad Hessian')


class TestLagrangePolyLinear(unittest.TestCase):
    def runTest(self):
        n = 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1)
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))

        xopt = model.xopt()
        for i in range(npt):
            c, g, hess = model.lagrange_polynomial(i)  # based at xopt
            for j in range(npt):
                dx = model.xpt(j) - xopt
                lag_value = c + model_value(g, hess, dx)
                expected_value = 1.0 if i==j else 0.0
                self.assertAlmostEqual(lag_value, expected_value, msg="Lagrange for x%g has bad value at x%g" % (i, j))


class TestLagrangePolyUnderdeterminedQuadratic(unittest.TestCase):
    def runTest(self):
        n = 2
        npt = n + 2
        x0 = np.array([1.0, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, objfun(x0), xl, xu, 1)
        x1 = x0 + np.array([1.0, 0.0])
        model.change_point(1, x1 - model.xbase, objfun(x1))
        x2 = x0 + np.array([0.1, 0.9])
        model.change_point(2, x2 - model.xbase, objfun(x2))
        x3 = x0 + np.array([-0.1, 0.0])
        model.change_point(3, x3 - model.xbase, objfun(x3))

        xopt = model.xopt()
        for i in range(npt):
            c, g, hess = model.lagrange_polynomial(i)  # based at xopt
            for j in range(npt):
                dx = model.xpt(j) - xopt
                lag_value = c + model_value(g, hess, dx)
                expected_value = 1.0 if i == j else 0.0
                self.assertAlmostEqual(lag_value, expected_value, msg="Lagrange for x%g has bad value at x%g" % (i, j))


class TestLagrangePolyUnderdeterminedQuadratic2(unittest.TestCase):
    def runTest(self):
        n = 2
        npt = 2 * n + 1
        x0 = np.array([1.0, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, objfun(x0), xl, xu, 1)
        x1 = x0 + np.array([1.0, 0.0])
        model.change_point(1, x1 - model.xbase, objfun(x1))
        x2 = x0 + np.array([0.1, 0.9])
        model.change_point(2, x2 - model.xbase, objfun(x2))
        x3 = x0 + np.array([-0.1, 0.0])
        model.change_point(3, x3 - model.xbase, objfun(x3))
        x4 = x0 + np.array([-0.1, 2.0])
        model.change_point(4, x4 - model.xbase, objfun(x4))

        xopt = model.xopt()
        for i in range(npt):
            c, g, hess = model.lagrange_polynomial(i)  # based at xopt
            for j in range(npt):
                dx = model.xpt(j) - xopt
                lag_value = c + model_value(g, hess, dx)
                expected_value = 1.0 if i == j else 0.0
                self.assertAlmostEqual(lag_value, expected_value, msg="Lagrange for x%g has bad value at x%g" % (i, j))


class TestLagrangePolyFullQuadratic(unittest.TestCase):
    def runTest(self):
        n = 2
        npt = (n + 1) * (n + 2) // 2
        x0 = np.array([1.0, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, objfun(x0), xl, xu, 1)
        x1 = x0 + np.array([1.0, 0.0])
        model.change_point(1, x1 - model.xbase, objfun(x1))
        x2 = x0 + np.array([0.1, 0.9])
        model.change_point(2, x2 - model.xbase, objfun(x2))
        x3 = x0 + np.array([-0.1, 0.0])
        model.change_point(3, x3 - model.xbase, objfun(x3))
        x4 = x0 + np.array([-0.1, 2.0])
        model.change_point(4, x4 - model.xbase, objfun(x4))
        x5 = x0 + np.array([-1.1, 1.0])
        model.change_point(5, x5 - model.xbase, objfun(x5))

        xopt = model.xopt()
        for i in range(npt):
            c, g, hess = model.lagrange_polynomial(i)  # based at xopt
            for j in range(npt):
                dx = model.xpt(j) - xopt
                lag_value = c + model_value(g, hess, dx)
                expected_value = 1.0 if i == j else 0.0
                self.assertAlmostEqual(lag_value, expected_value, msg="Lagrange for x%g has bad value at x%g" % (i, j))


class TestPoisednessLinear(unittest.TestCase):
    def runTest(self):
        n = 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        delta = 0.5
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1)
        model.add_new_sample(0, rosenbrock(x0))
        x1 = x0 + delta * np.array([1.0, 0.0])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = x0 + delta * np.array([0.0, 1.0])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        model.kopt = 0  # force this
        # Here (use delta=1), Lagrange polynomials are (1-x-y), 1-x and 1-y
        # Maximum value in ball is for (1-x-y) at (x,y)=(1/sqrt2, 1/sqrt2) --> max value = 1 + sqrt(2)
        self.assertAlmostEqual(model.poisedness_constant(delta), 1.0 + sqrt(2.0), places=6, msg="Poisedness wrong")


class TestPoisednessFullQuadratic(unittest.TestCase):
    def runTest(self):
        # DFO book, Figure 3.1 (note errata) - solution from Mathematica
        n = 2
        npt = (n + 1) * (n + 2) // 2
        x0 = np.array([0.5, 0.5])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, objfun(x0), xl, xu, 1)
        x1 = np.array([0.05, 0.1])
        model.change_point(1, x1 - model.xbase, objfun(x1))
        x2 = np.array([0.1, 0.05])
        model.change_point(2, x2 - model.xbase, objfun(x2))
        x3 = np.array([0.95, 0.9])
        model.change_point(3, x3 - model.xbase, objfun(x3))
        x4 = np.array([0.9, 0.95])
        model.change_point(4, x4 - model.xbase, objfun(x4))
        x5 = np.array([0.85, 0.85])
        model.change_point(5, x5 - model.xbase, objfun(x5))
        delta = 0.5
        model.kopt = 0  # force base point
        self.assertAlmostEqual(model.poisedness_constant(delta), 294.898, places=2, msg="Poisedness wrong")


class TestPoisednessUnderdeterminedQuadratic(unittest.TestCase):
    def runTest(self):
        # Based originally on DFO book, Figure 3.3 - solution from Mathematica
        n = 2
        npt = 2*n + 1
        x0 = np.array([0.5, 0.5])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, objfun(x0), xl, xu, 1)
        x1 = np.array([0.524, 0.0006])
        model.change_point(1, x1 - model.xbase, objfun(x1))
        x2 = np.array([0.032, 0.323])
        model.change_point(2, x2 - model.xbase, objfun(x2))
        x3 = np.array([0.187, 0.89])
        model.change_point(3, x3 - model.xbase, objfun(x3))
        x4 = np.array([0.982, 0.368])
        model.change_point(4, x4 - model.xbase, objfun(x4))
        delta = 0.5
        model.kopt = 0  # force base point
        self.assertAlmostEqual(model.poisedness_constant(delta), 1.10018, places=3, msg="Poisedness wrong")


class TestAddPoint(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, 1)
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        # Now add a new point
        x3 = np.array([1.0, 1.0])  # good point
        add_ok = model.add_new_point(x3 - model.xbase, rosenbrock(x3))
        self.assertTrue(add_ok, "Adding x3 failed")
        self.assertEqual(model.npt(), 4, "Wrong number of points after x3")
        self.assertTrue(array_compare(model.xpt(3, abs_coordinates=True), x3), "Wrong new point after x3")
        self.assertTrue(array_compare(model.fval(3), rosenbrock(x3)), "Wrong fval after x3")
        self.assertEqual(model.kopt, 3, "Wrong kopt after x3")
        self.assertEqual(len(model.nsamples), 4, "Wrong nsamples length after x3")
        self.assertEqual(model.nsamples[-1], 1, "Wrong nsample value after x3")
        x4 = np.array([-1.8, 1.8])  # bad point
        add_ok = model.add_new_point(x4 - model.xbase, rosenbrock(x4))
        self.assertTrue(add_ok, "Adding x4 failed")
        self.assertEqual(model.npt(), 5, "Wrong number of points after x4")
        self.assertTrue(array_compare(model.xpt(4, abs_coordinates=True), x4), "Wrong new point after x4")
        self.assertTrue(array_compare(model.fval(4), rosenbrock(x4)), "Wrong fval after x4")
        self.assertEqual(model.kopt, 3, "Wrong kopt after x4")
        x5 = np.array([-1.0, 1.0])
        add_ok = model.add_new_point(x5 - model.xbase, rosenbrock(x5))
        self.assertTrue(add_ok, "Adding x5 failed")
        self.assertEqual(model.npt(), 6, "Wrong number of points after x5")
        x6 = np.array([-1.5, 1.5])
        add_ok = model.add_new_point(x6 - model.xbase, rosenbrock(x6))
        self.assertFalse(add_ok, "Adding x6 should have failed")
        self.assertEqual(model.npt(), 6, "Wrong number of points after x6")
        self.assertTrue(array_compare(model.xpt(5, abs_coordinates=True), x5), "Wrong new point after x6")
        self.assertTrue(array_compare(model.fval(5), rosenbrock(x5)), "Wrong fval after x6")
        self.assertEqual(model.kopt, 3, "Wrong kopt after x6")
