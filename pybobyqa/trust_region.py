"""
Trust Region Subproblem Solver
====

Specifically, the call
    d, gnew, crvmin = trsbox(xopt, g, H, sl, su, delta)
produces a new vector d which (approximately) solves the trust region subproblem:
    min_{d}  g'*d + 0.5*d'*H*d
    s.t.    ||d|| <= delta
            sl <= xopt + d <= su
The other outputs: gnew is the gradient of the model at d, and crvmin has
information about the curvature of the model at the solution:
- If ||d||=delta, then crvmin = 0
- If d is constrained in all directions by the box constraints, then crvmin = -1
- Otherwise, crvmin > 0 is the smallest positive curvature seen in the Hessian, min_{k} (dk^T H dk / ||dk||^2) for iterates dk

For problems with general convex constraints, we have an alternative function
    d, gnew, crvmin = ctrsbox(xbase, xopt, g, H, sl, su, projections, delta)
which solves the subproblem
    min_{d}  g'*d + 0.5*d'*H*d
    s.t.    ||d|| <= delta
            sl <= xopt + d <= su
            xbase + xopt + d in the intersection of all convex sets C with proj_{C} in the list 'projections'
The outputs are the same as trsbox, but without the negative case for crvmin.
This version is solved using projected gradient descent; see Chapter 10 of
A. Beck, First-Order Methods in Optimization. SIAM, 2017.

Notes
----
The solver trsbox is an implementation of the routine TRSBOX from BOBYQA (Powell, 2009).
Some modifications to the termination conditions are from the equivalent routine
from DFBOLS (Zhang et al, 2010).


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

from math import sqrt
import numpy as np
import warnings

try:
    import trustregion
    USE_FORTRAN = True
except ImportError:
    # Fall back to Python implementation
    USE_FORTRAN = False


from .util import sumsq, model_value, dykstra, pball, pbox


__all__ = ['trsbox', 'trsbox_geometry', 'ctrsbox', 'ctrsbox_geometry']

ZERO_THRESH = 1e-14


def trsbox(xopt, g, H, sl, su, delta, use_fortran=USE_FORTRAN):
    if use_fortran:
        return trustregion.solve(g, H, delta,
                                 sl=np.minimum(sl - xopt, -ZERO_THRESH),
                                 su=np.maximum(su - xopt, ZERO_THRESH),
                                 verbose_output=True)

    n = xopt.size
    assert xopt.shape == (n,), "xopt has wrong shape (should be vector)"
    assert g.shape == (n,), "g and xopt have incompatible sizes"
    assert len(H.shape) == 2, "H must be a matrix"
    assert H.shape == (n,n), "H and xopt have incompatible sizes"
    if not np.allclose(H, H.T):
        # Enforce symmetry
        H = 0.5 * (H + H.T)
        warnings.warn("Trust-region solver: fixing non-symmetric Hessian", RuntimeWarning)
    assert sl.shape == (n,), "sl and xopt have incompatible sizes"
    assert su.shape == (n,), "su and xopt have incompatible sizes"
    assert delta > 0.0, "delta must be strictly positive"
    # Assume g and H have full quadratic model for objective
    # i.e. skip straight to label 8 in DFBOLS version

    # The sign of G(I) gives the sign of the change to the I-th variable
    # that will reduce Q from its value at XOPT. Thus XBDI(I) shows whether
    # or not to fix the I-th variable at one of its bounds initially, with
    # NACT being set to the number of fixed variables. D and GNEW are also
    # set for the first iteration. DELSQ is the upper bound on the sum of
    # squares of the free variables. QRED is the reduction in Q so far.

    iterc = 0
    nact = 0  # number of fixed variables

    xbdi = np.zeros((n,), dtype=int)  # fix x_i at bounds? [values -1, 0, 1]
    xbdi[(xopt <= sl) & (g >= 0.0)] = -1
    xbdi[(xopt >= su) & (g <= 0.0)] = 1

    d = np.zeros((n,))
    s = np.zeros((n,))
    gnew = g.copy()
    qred = 0.0
    delsq = delta ** 2
    crvmin = -1.0
    beta = 0.0  # label 20

    need_alt_trust_step = False  # will either quit main CG loop to finish, or do alternative step
    MAX_LOOP_ITERS = 100 * n ** 2  # avoid infinite loops
    # while True:  # main CG loop [label 30]
    for ii in range(MAX_LOOP_ITERS):
        s[xbdi != 0] = 0.0
        if beta == 0.0:
            s[xbdi == 0] = -gnew[xbdi == 0]
        else:
            s[xbdi == 0] = beta * s[xbdi == 0] - gnew[xbdi == 0]
        stepsq = sumsq(s)

        if stepsq == 0.0:
            need_alt_trust_step = False
            break  # break and quit

        if beta == 0.0:
            gredsq = stepsq
            itermax = iterc + n - nact

        if iterc == 0:
            gredsq0 = gredsq

        # Exit conditions
        if gredsq <= min(1.0e-6 * gredsq0, 1.0e-18) or gredsq * delsq <= min(1.0e-6 * qred ** 2, 1.0e-18):  # DFBOLS
            need_alt_trust_step = False
            break  # break and quit

        # Multiply the search direction by the second derivative matrix of Q and
        # calculate some scalars for the choice of steplength. Then set BLEN to
        # the length of the the step to the trust region boundary and STPLEN to
        # the steplength, ignoring the simple bounds.

        hs = H.dot(s)

        # label 50
        ds = np.dot(s[xbdi == 0], d[xbdi == 0])
        shs = np.dot(s[xbdi == 0], hs[xbdi == 0])
        resid = delsq - sumsq(d[xbdi == 0])
        if resid <= 0.0:
            need_alt_trust_step = True
            break  # break and calculate alt step instead

        temp = sqrt(stepsq * resid + ds ** 2)
        blen = (resid / (temp + ds) if ds >= 0.0 else (temp - ds) / stepsq)
        stplen = (blen if shs <= 0.0 else min(blen, gredsq / shs))

        # Exit condition
        if stplen <= 1.0e-30:  # DFBOLS
            need_alt_trust_step = False
            break  # break and quit

        # Reduce STPLEN if necessary in order to preserve the simple bounds,
        # letting IACT be the index of the new constrained variable.
        iact = None
        for i in range(n):
            if s[i] != 0.0:
                temp = (su[i] - xopt[i] - d[i] if s[i] > 0.0 else sl[i] - xopt[i] - d[i]) / s[i]
                if temp < stplen:
                    stplen = temp
                    iact = i

        # Update CRVMIN, GNEW and D. Set SDEC to the decrease that occurs in Q.
        sdec = 0.0
        if stplen > 0.0:
            iterc += 1
            temp = shs / stepsq
            if iact is None and temp > 0.0:
                crvmin = min(crvmin, temp) if crvmin != -1.0 else temp
            ggsav = gredsq
            gnew += stplen * hs
            d += stplen * s
            gredsq = sumsq(gnew[xbdi == 0])
            sdec = max(stplen * (ggsav - 0.5 * stplen * shs), 0.0)
            qred += sdec

        # Restart the conjugate gradient method if it has hit a new bound.
        if iact is not None:
            nact += 1
            xbdi[iact] = (1 if s[iact] >= 0.0 else -1)
            delsq = delsq - d[iact] ** 2
            if delsq <= 0.0:
                need_alt_trust_step = True
                break  # break and calculate alt step instead
            beta = 0.0  # label 20
            continue  # restart loop (new CG iteration)

        # If STPLEN is less than BLEN, then either apply another conjugate
        # gradient iteration or RETURN.
        if stplen >= blen:
            need_alt_trust_step = True
            break  # break and calculate alt step instead

        # Exit condition
        if iterc == itermax or sdec <= 1.0e-6 * qred:  # DFBOLS
            need_alt_trust_step = False
            break  # break and quit

        beta = gredsq / ggsav
        continue  # new CG iteration
    # end of CG loop

    # either done or need to take and alternative step
    if need_alt_trust_step:
        crvmin = 0.0
        d, gnew = alt_trust_step(n, xopt, H, sl, su, d, xbdi, nact, gnew, qred)
        return d, gnew, crvmin
    else:
        return d_within_bounds(d, xopt, sl, su, xbdi), gnew, crvmin


# Alternative Trust Region Step (label 100 of TRSBOX in BOBYQA, where crvmin=0)
def alt_trust_step(n, xopt, H, sl, su, d, xbdi, nact, gnew, qred):
    MAX_LOOP_ITERS = 100 * n ** 2  # avoid infinite loops
    # while True:  # label 100 here
    for ii in range(MAX_LOOP_ITERS):
        if nact >= n - 1:
            return d_within_bounds(d, xopt, sl, su, xbdi), gnew

        # Prepare for the alternative iteration by calculating some scalars
        # and by multiplying the reduced D by the second derivative matrix of
        # Q, where S holds the reduced D in the call of GGMULT.
        s = np.zeros((n,))
        s[xbdi == 0] = d[xbdi == 0]
        dredsq = sumsq(d[xbdi == 0])
        dredg = np.dot(d[xbdi == 0], gnew[xbdi == 0])
        gredsq = sumsq(gnew[xbdi == 0])

        # Label 210 (crvmin = 0, itcsav = iterc)
        hs = H.dot(s)

        hred = hs.copy()
        # quit 210 by goto 120

        # Let the search direction S be a linear combination of the reduced D
        # and the reduced G that is orthogonal to the reduced D.
        restart_alt_loop = False  # once the below loop finishes, quit unless need to go again
        # while True:  # label 120
        for jj in range(MAX_LOOP_ITERS):
            temp = gredsq * dredsq - dredg ** 2
            if temp <= 1.0e-4 * qred ** 2:
                restart_alt_loop = False
                break  # quit inner label 120 loop and return results
            temp = sqrt(temp)
            s = np.zeros((n,))
            s[xbdi == 0] = (dredg * d[xbdi == 0] - dredsq * gnew[xbdi == 0]) / temp
            sredg = -temp

            # By considering the simple bounds on the variables, calculate an upper
            # bound on the tangent of half the angle of the alternative iteration,
            # namely ANGBD, except that, if already a free variable has reached a
            # bound, there is a branch back to label 100 after fixing that variable.
            free_variable_reached_bound = False
            angbd = 1.0
            iact = None
            for i in range(n):
                if xbdi[i] == 0:
                    tempa = xopt[i] + d[i] - sl[i]
                    tempb = su[i] - xopt[i] - d[i]
                    if tempa <= 0.0:
                        nact += 1
                        xbdi[i] = -1
                        free_variable_reached_bound = True
                        break  # skip the rest of this for loop
                    elif tempb <= 0.0:
                        nact += 1
                        xbdi[i] = 1
                        free_variable_reached_bound = True
                        break  # skip the rest of this for loop
                    ssq = d[i] ** 2 + s[i] ** 2
                    temp = ssq - (xopt[i] - sl[i]) ** 2
                    if temp > 0.0:
                        temp = sqrt(temp) - s[i]
                        if angbd * temp > tempa:
                            angbd = tempa / temp
                            iact = i
                            xsav = -1
                    temp = ssq - (su[i] - xopt[i]) ** 2
                    if temp > 0.0:
                        temp = sqrt(temp) + s[i]
                        if angbd * temp > tempb:
                            angbd = tempb / temp
                            iact = i
                            xsav = 1
            # End for loop
            if free_variable_reached_bound:  # deal with break conditions above
                restart_alt_loop = True
                break  # quit inner label 120 loop and restart alt iteration loop (label 100)

            # Label 210 (crvmin = 0, itcsav < iterc since iterc+=1 earlier)
            hs = H.dot(s)

            # Label 150
            # Calculate HHD and some curvatures for the alternative iteration.
            shs = np.sum(s[xbdi == 0] * hs[xbdi == 0])
            dhs = np.sum(d[xbdi == 0] * hs[xbdi == 0])
            dhd = np.sum(d[xbdi == 0] * hred[xbdi == 0])

            # Seek the greatest reduction in Q for a range of equally spaced values
            # of ANGT in [0,ANGBD], where ANGT is the tangent of half the angle of
            # the alternative iteration.
            redmax = 0.0
            isav = -1
            redsav = 0.0
            temp = 0.0  # force scope outside i loop below since needed later
            iu = int(17 * angbd + 3.1)
            for i in range(iu):  # i = 0, ..., iu-1
                angt = angbd * float(i + 1) / float(iu)
                sth = 2.0 * angt / (1.0 + angt ** 2)
                temp = shs + angt * (angt * dhd - 2.0 * dhs)
                rednew = sth * (angt * dredg - sredg - 0.5 * sth * temp)
                if rednew > redmax:
                    redmax = rednew
                    isav = i
                    rdprev = redsav
                elif i == isav + 1:
                    rdnext = rednew
                redsav = rednew

            # Return if the reduction is zero. Otherwise, set the sine and cosine
            # of the angle of the alternative iteration, and calculate SDEC.
            if isav == -1:
                restart_alt_loop = False
                break  # quit inner label 120 loop and return results

            if isav < iu - 1:
                temp = (rdnext - rdprev) / (2.0 * redmax - rdprev - rdnext)
                angt = angbd * (float(isav + 1) + 0.5 * temp) / float(iu)

            cth = (1.0 - angt ** 2) / (1.0 + angt ** 2)
            sth = 2.0 * angt / (1.0 + angt ** 2)
            temp = shs + angt * (angt * dhd - 2.0 * dhs)
            sdec = sth * (angt * dredg - sredg - 0.5 * sth * temp)

            if sdec <= 0.0:
                restart_alt_loop = False
                break  # quit inner label 120 loop and return results

            # Update GNEW, D and HRED. If the angle of the alternative iteration
            # is restricted by a bound on a free variable, that variable is fixed
            # at the bound.
            gnew += (cth - 1.0) * hred + sth * hs
            d[xbdi == 0] = cth * d[xbdi == 0] + sth * s[xbdi == 0]
            dredg = np.dot(d[xbdi == 0], gnew[xbdi == 0])
            gredsq = sumsq(gnew[xbdi == 0])
            hred = cth * hred + sth * hs

            qred += sdec
            if iact is not None and isav == iu - 1:
                nact += 1
                xbdi[iact] = xsav
                restart_alt_loop = True
                break  # quit inner label 120 loop and restart alt iteration loop (label 100)

            if (sdec <= 0.01 * qred):
                restart_alt_loop = False
                break  # quit inner label 120 loop and return results
            continue  # back to inner label 120 loop

        # End inner label 120 loop

        if restart_alt_loop:
            continue
        else:
            break  # end outer loop and quit

    # End while True (label 100)
    return d_within_bounds(d, xopt, sl, su, xbdi), gnew


def d_within_bounds(d, xopt, sl, su, xbdi):
    # Used in TRSBOX, force d to be within bounds
    # In Fortran code, is at label 190
    xnew = np.maximum(np.minimum(xopt + d, su), sl)
    xnew[xbdi == -1] = sl[xbdi == -1]
    xnew[xbdi == 1] = su[xbdi == 1]
    d = xnew - xopt
    return d


def trsbox_geometry(xbase, c, g, H, lower, upper, Delta, use_fortran=USE_FORTRAN):
    # Given a Lagrange polynomial defined by: L(x) = c + g' * (x - xbase) + 0.5*(x-xbase)*H*(x-xbase)
    # Maximise |L(x)| in a box + trust region - that is, solve:
    #   max_x  abs(c + g' * (x - xbase) + 0.5*(x-xbase)*H*(x-xbase))
    #    s.t.  lower <= x <= upper
    #          ||x-xbase|| <= Delta
    # Setting s = x-xbase (or x = xbase + s), this is equivalent to:
    #   max_s  abs(c + g' * s + 0.5*s*H*s)
    #   s.t.   lower <= xbase + s <= upper
    #          ||s|| <= Delta
    smin, gmin, crvmin = trsbox(xbase, g, H, lower, upper, Delta, use_fortran=use_fortran)  # minimise L(x)
    smax, gmax, crvmax = trsbox(xbase, -g, -H, lower, upper, Delta, use_fortran=use_fortran)  # maximise L(x)
    if abs(c + model_value(g, H, smin)) >= abs(c + model_value(g, H, smax)):  # take largest abs value
        return xbase + smin
    else:
        return xbase + smax


def ctrsbox(xbase, xopt, g, H, sl, su, projections, delta, dykstra_max_iters=100, dykstra_tol=1e-10, gtol=1e-8):
    if projections is None or len(projections) == 0:
        return trsbox(xopt, g, H, sl, su, delta)

    n = xopt.size
    assert xopt.shape == (n,), "xopt has wrong shape (should be vector)"
    assert xbase.shape == (n,), "xbase and xopt has wrong shape (should be vector)"
    assert g.shape == (n,), "g and xopt have incompatible sizes"
    assert len(H.shape) == 2, "H must be a matrix"
    assert H.shape == (n, n), "H and xopt have incompatible sizes"
    if not np.allclose(H, H.T):
        # Enforce symmetry
        H = 0.5 * (H + H.T)
        warnings.warn("Trust-region solver: fixing non-symmetric Hessian", RuntimeWarning)
    assert sl.shape == (n,), "sl and xopt have incompatible sizes"
    assert su.shape == (n,), "su and xopt have incompatible sizes"
    assert delta > 0.0, "delta must be strictly positive"

    # Get full list of required projections (i.e. including TR constraint and box constraints)
    proj_tr = lambda w: pball(w, xbase + xopt, delta)
    proj_box = lambda w: pbox(w, xbase + sl, xbase + su)
    P = list(projections)  # make a copy of the projections list
    P.append(proj_tr)
    P.append(proj_box)

    # Compute the joint projection into all constraints, but using increments from xopt
    def proj(d0):
        p = dykstra(P, xbase + xopt + d0, max_iter=dykstra_max_iters, tol=dykstra_tol)
        # we want the step only, so we subtract xbase+xopt
        # from the new point: proj(xk+d) - xk
        return p - xopt - xbase

    # Compute Lipschitz constant of quadratic objective, L >= ||H||_2
    # Stepsize for projected GD will be Lk=L
    L = np.linalg.norm(H)  # use L = ||H||_F >= ||H|_2 as valid L-smooth constant
    L = max(L, delta / np.linalg.norm(g))  # another upper bound; if L~0 then will take steps (1/L)*g, and don't want to go beyond trust-region

    MAX_LOOP_ITERS = 100 * n ** 2

    d = np.zeros((n,))
    gnew = g.copy()
    crvmin = -1.0

    # projected GD loop
    for ii in range(MAX_LOOP_ITERS):
        s = proj(d - (1 / L) * gnew) - d  # new step is dnew = d + s

        # take the step
        d += s
        gnew += H.dot(s)

        # update CRVMIN
        crv = np.dot(H.dot(s), s) / sumsq(s) if sumsq(s) >= ZERO_THRESH else crvmin
        crvmin = min(crvmin, crv) if crvmin != -1.0 else crv

        # exit condition
        if np.linalg.norm(s) <= gtol:
            break

    # Lastly, ensure the explicit bounds are hard-enforced
    if np.linalg.norm(d) > delta:
        d *= delta / np.linalg.norm(d)
    d = np.maximum(np.minimum(xopt + d, su), sl) - xopt  # does not increase ||d||
    gnew = g + H.dot(d)

    if np.linalg.norm(d) >= delta - ZERO_THRESH:
        crvmin = 0.0
    return d, gnew, crvmin


def ctrsbox_geometry(xbase, xopt, c, g, H, lower, upper, projections, Delta, dykstra_max_iters=100, dykstra_tol=1e-10, gtol=1e-8):
    # Given a Lagrange polynomial defined by: L(x) = c + g' * (x - xopt) + 0.5*(x-xopt)*H*(x-xopt)
    # Maximise |L(x)| in the feasible region + trust region - that is, solve:
    #   max_x  abs(c + g' * (x - xopt) + 0.5*(x-xopt)*H*(x-xopt))
    #    s.t.  lower <= x <= upper
    #          ||x-xopt|| <= Delta
    #          xbase+x in the intersection of all convex sets C with proj_{C} in the list 'projections'
    # Setting s = x-xbase (or x = xopt + s), this is equivalent to:
    #   max_s  abs(c + g' * s + 0.5*s*H*s)
    #   s.t.   lower <= xopt + s <= upper
    #          ||s|| <= Delta
    #          xbase + xopt + s in the intersection of all convex sets C with proj_{C} in the list 'projections'
    if projections is None or len(projections) == 0:
        return trsbox_geometry(xopt, g, H, lower, upper, Delta)

    smin, gmin, crvmin = ctrsbox(xbase, xopt, g, H, lower, upper, projections, Delta,
                                 dykstra_max_iters=dykstra_max_iters, dykstra_tol=dykstra_tol, gtol=gtol)  # minimise L(x)
    smax, gmax, crvmax = ctrsbox(xbase, xopt, -g, -H, lower, upper, projections, Delta,
                                 dykstra_max_iters=dykstra_max_iters, dykstra_tol=dykstra_tol, gtol=gtol)  # maximise L(x)
    if abs(c + model_value(g, H, smin)) >= abs(c + model_value(g, H, smax)):  # take largest abs value
        return xopt + smin
    else:
        return xopt + smax