"""
Solver
====

The main solver


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
import scipy.stats as STAT
import warnings

from .controller import *
from .diagnostic_info import *
from .params import *
from .util import *

__all__ = ['solve']


# A container for the results of the optimization routine
class OptimResults(object):
    def __init__(self, xmin, fmin, gradmin, hessmin, nf, nx, nruns, exit_flag, exit_msg):
        self.x = xmin
        self.f = fmin
        self.gradient = gradmin
        self.hessian = hessmin
        self.nf = nf
        self.nx = nx
        self.nruns = nruns
        self.flag = exit_flag
        self.msg = exit_msg
        self.diagnostic_info = None
        # Set standard names for exit flags
        self.EXIT_SLOW_WARNING = EXIT_SLOW_WARNING
        self.EXIT_MAXFUN_WARNING = EXIT_MAXFUN_WARNING
        self.EXIT_SUCCESS = EXIT_SUCCESS
        self.EXIT_INPUT_ERROR = EXIT_INPUT_ERROR
        self.EXIT_TR_INCREASE_ERROR = EXIT_TR_INCREASE_ERROR
        self.EXIT_LINALG_ERROR = EXIT_LINALG_ERROR
        self.EXIT_FALSE_SUCCESS_WARNING = EXIT_FALSE_SUCCESS_WARNING

    def __str__(self):
        # Result of calling print(soln)
        output = "****** Py-BOBYQA Results ******\n"
        if self.flag != self.EXIT_INPUT_ERROR:
            output += "Solution xmin = %s\n" % str(self.x)
            output += "Objective value f(xmin) = %.10g\n" % self.f
            output += "Needed %g objective evaluations (at %g points)\n" % (self.nf, self.nx)
            if self.nruns > 1:
                output += "Did a total of %g runs\n" % self.nruns
            if self.gradient is not None and np.size(self.gradient) < 100:
                output += "Approximate gradient = %s\n" % str(self.gradient)
            elif self.gradient is None:
                output += "No gradient available\n"
            else:
                output += "Not showing approximate gradient because it is too long; check self.gradient\n"
            if self.hessian is not None and np.size(self.hessian) < 200:
                output += "Approximate Hessian = %s\n" % str(self.hessian)
            elif self.hessian is None:
                output += "No Hessian available\n"
            else:
                output += "Not showing approximate Hessian because it is too long; check self.hessian\n"
            if self.diagnostic_info is not None:
                output += "Diagnostic information available; check self.diagnostic_info\n"
        output += "Exit flag = %g\n" % self.flag
        output += "%s\n" % self.msg
        output += "******************************\n"
        return output


def solve_main(objfun, x0, args, xl, xu, npt, rhobeg, rhoend, maxfun, nruns_so_far, nf_so_far, nx_so_far, nsamples, params,
               diagnostic_info, scaling_changes, f0_avg_old=None, f0_nsamples_old=None):
    # Evaluate at x0 (keep nf, nx correct and check for f small)
    if f0_avg_old is None:
        number_of_samples = max(nsamples(rhobeg, rhobeg, 0, nruns_so_far), 1)
        # Evaluate the first time...
        nf = nf_so_far + 1
        nx = nx_so_far + 1
        f0 = eval_objective(objfun, remove_scaling(x0, scaling_changes), args, eval_num=nf, pt_num=nx,
                                                full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                              check_for_overflow=params("general.check_objfun_for_overflow"))

        # Now we have m, we can evaluate the rest of the times
        f_list = np.zeros((number_of_samples,))
        f_list[0] = f0
        num_samples_run = 1
        exit_info = None

        for i in range(1, number_of_samples):  # skip first eval - already did this
            if nf >= maxfun:
                exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                nruns_so_far += 1
                break  # stop evaluating at x0

            nf += 1
            # Don't increment nx for x0 - we did this earlier
            f_list[i] = eval_objective(objfun, remove_scaling(x0, scaling_changes), args, eval_num=nf, pt_num=nx,
                                                full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                                check_for_overflow=params("general.check_objfun_for_overflow"))
            num_samples_run += 1

        f0_avg = np.mean(f_list[:num_samples_run])
        if f0_avg <= params("model.abs_tol"):
            exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")

        if exit_info is not None:
            return x0, f0_avg, None, None, num_samples_run, nf, nx, nruns_so_far+1, exit_info, diagnostic_info

    else:  # have old f0 information (e.g. from previous restart), use this instead

        f0_avg = f0_avg_old
        num_samples_run = f0_nsamples_old
        nf = nf_so_far
        nx = nx_so_far

    # Initialise controller
    control = Controller(objfun, x0, args, f0_avg, num_samples_run, xl, xu, npt, rhobeg, rhoend, nf, nx, maxfun, params, scaling_changes)

    # Initialise interpolation set
    number_of_samples = max(nsamples(control.delta, control.rho, 0, nruns_so_far), 1)
    num_directions = npt - 1
    if params("init.random_initial_directions"):
        logging.info("Initialising (random directions)")
        exit_info = control.initialise_random_directions(number_of_samples, num_directions, params)
    else:
        logging.info("Initialising (coordinate directions)")
        exit_info = control.initialise_coordinate_directions(number_of_samples, num_directions, params)
    if exit_info is not None:
        x, f, gradmin, hessmin, nsamples = control.model.get_final_results()
        return x, f, None, None, nsamples, control.nf, control.nx, nruns_so_far + 1, exit_info, diagnostic_info

    # Save list of last N successful steps: whether they failed to be an improvement over fsave
    succ_steps_not_improvement = [False]*params("restarts.soft.max_fake_successful_steps")

    # Attempting to auto-detect restart? Need to keep a history of delta and ||chg J|| for non-safety iterations
    restart_auto_detect_full = False  # have we filled up the whole vectors yet? Don't restart from this if not
    if params("restarts.use_restarts") and params("restarts.auto_detect"):
        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
        restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
        restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))

    #------------------------------------------
    # Begin main loop
    # ------------------------------------------
    current_iter = -1
    logging.info("Beginning main loop")
    while True:
        current_iter += 1

        # Noise level exit check
        if params("noise.quit_on_noise_level") and control.all_values_within_noise_level(params):
            if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                     x_in_abs_coords_to_save=None, f_to_save=None, nsamples_to_save=None)
                if exit_info is not None:
                    nruns_so_far += 1
                    break  # quit
                current_iter = -1
                nruns_so_far += 1
                rhoend = params("restarts.rhoend_scale") * rhoend
                restart_auto_detect_full = False
                restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                continue  # next iteration
            else:
                exit_info = ExitInformation(EXIT_SUCCESS, "All points within noise level")
                nruns_so_far += 1
                break  # quit

        interp_ok, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error = \
            control.model.interpolate_model(verbose=params("logging.save_diagnostic_info"),
                min_chg_hess=params("interpolation.minimum_change_hessian"),
                get_norm_model_chg=params("restarts.use_restarts") and params("restarts.auto_detect"))
        if not interp_ok:
            if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                 x_in_abs_coords_to_save=None, f_to_save=None, nsamples_to_save=None)
                if exit_info is not None:
                    nruns_so_far += 1
                    break  # quit
                current_iter = -1
                nruns_so_far += 1
                rhoend = params("restarts.rhoend_scale") * rhoend
                restart_auto_detect_full = False
                restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                continue  # next iteration
            else:
                exit_info = ExitInformation(EXIT_LINALG_ERROR, "Singular matrix in mini-model interpolation (main loop)")
                nruns_so_far += 1
                break  # quit


        # Trust region step
        d, gopt, hq, gnew, crvmin = control.trust_region_step()
        logging.debug("Trust region step is d = " + str(d))
        xnew = control.model.xopt() + d
        dnorm = min(LA.norm(d), control.delta)

        if params("logging.save_diagnostic_info"):
            diagnostic_info.save_info_from_control(control, nruns_so_far, current_iter,
                                                   save_poisedness=params("logging.save_poisedness"))
            diagnostic_info.update_interpolation_information(interp_error, interp_cond_num, norm_chg_grad,
                                                             norm_chg_hess, LA.norm(gopt), LA.norm(d))

        if dnorm < params("general.safety_step_thresh") * control.rho:
            # (start safety step)
            logging.debug("Safety step (main phase)")

            if params("logging.save_diagnostic_info"):
                diagnostic_info.update_ratio(np.nan)
                diagnostic_info.update_iter_type(ITER_SAFETY)
                diagnostic_info.update_slow_iter(-1)

            if not control.done_with_current_rho(xnew, gnew, crvmin, hq, current_iter):
                distsq = (10.0 * control.rho) ** 2
                number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                update_delta = True  # we do reduce delta for safety steps
                did_fix_geom, exit_info = control.check_and_fix_geometry(distsq, update_delta, number_of_samples, params)
                if dnorm > control.rho:
                    control.last_successful_iter = current_iter

                if exit_info is not None:
                    if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                            "restarts.use_soft_restarts"):
                        number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                        exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                         x_in_abs_coords_to_save=None, f_to_save=None,
                                                         nsamples_to_save=None)
                        if exit_info is not None:
                            nruns_so_far += 1
                            break  # quit
                        current_iter = -1
                        nruns_so_far += 1
                        rhoend = params("restarts.rhoend_scale") * rhoend
                        restart_auto_detect_full = False
                        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        continue  # next iteration
                    else:
                        nruns_so_far += 1
                        break  # quit

                if did_fix_geom:
                    continue  # next iteration

            # If we are done with the current rho, or didn't fix geometry above, reduce rho
            if control.rho > rhoend:
                # Reduce rho
                control.reduce_rho(current_iter, params)
                logging.info("New rho = %g after %i function evaluations" % (control.rho, control.nf))
                if control.n() < params("logging.n_to_print_whole_x_vector"):
                    logging.debug("Best so far: f = %.15g at x = " % (control.model.fopt())
                                  + str(control.model.xopt(abs_coordinates=True)))
                else:
                    logging.debug("Best so far: f = %.15g at x = [...]" % (control.model.fopt()))
                continue  # next iteration
            else:
                # Quit on rho=rhoend
                if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                         x_in_abs_coords_to_save=None, f_to_save=None, nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    # Cannot reduce rho, so check xnew and quit
                    x = control.model.as_absolute_coordinates(xnew)
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    f_list, num_samples_run, exit_info = control.evaluate_objective(x, number_of_samples,
                                                                                               params)
                    
                    if num_samples_run > 0:
                            control.model.save_point(x, np.mean(f_list[:num_samples_run]), num_samples_run, x_in_abs_coords=True)
                    
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit

                    exit_info = ExitInformation(EXIT_SUCCESS, "rho has reached rhoend")
                    nruns_so_far += 1
                    break  # quit
            # (end safety step)
        else:
            # (start trust region step)
            logging.debug("Standard trust region step")

            # Add chgJ and delta to restart auto-detect set
            if params("restarts.use_restarts") and params("restarts.auto_detect"):
                if restart_auto_detect_full:
                    # Drop first values, add new values at end
                    restart_auto_detect_delta = np.append(np.delete(restart_auto_detect_delta, [0]), control.delta)
                    restart_auto_detect_chg_grad = np.append(np.delete(restart_auto_detect_chg_grad, [0]), norm_chg_grad)
                    restart_auto_detect_chg_hess = np.append(np.delete(restart_auto_detect_chg_hess, [0]), norm_chg_hess)
                else:
                    idx = np.argmax(restart_auto_detect_delta < 0.0)  # index of first negative value
                    restart_auto_detect_delta[idx] = control.delta
                    restart_auto_detect_chg_grad[idx] = norm_chg_grad
                    restart_auto_detect_chg_hess[idx] = norm_chg_hess
                    restart_auto_detect_full = (idx >= len(restart_auto_detect_delta) - 1)  # have we now got everything?

            if sumsq(d) <= params("general.rounding_error_constant") * sumsq(control.model.xopt()):
                base_shift = control.model.xopt()
                xnew = xnew - base_shift  # before xopt is updated
                control.model.shift_base(base_shift)

            knew, exit_info = control.choose_point_to_replace(d, skip_kopt=True)
            if exit_info is not None:
                if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                        "restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, f_to_save=None,
                                                     nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    nruns_so_far += 1
                    break  # quit

            # Evaluate new point
            x = control.model.as_absolute_coordinates(xnew)
            number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
            f_list, num_samples_run, exit_info = control.evaluate_objective(x, number_of_samples, params)
            if exit_info is not None:
                if num_samples_run > 0:
                    control.model.save_point(x, np.mean(f_list[:num_samples_run]), num_samples_run, x_in_abs_coords=True)
                nruns_so_far += 1
                break  # quit

            # Estimate f in order to compute 'actual reduction'
            ratio, exit_info = control.calculate_ratio(current_iter, f_list[:num_samples_run], d, gopt, hq)
            if exit_info is not None:
                if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                        "restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, f_to_save=None,
                                                     nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    nruns_so_far += 1
                    break  # quit

            # Update delta
            logging.debug("Ratio = %g" % ratio)
            if params("logging.save_diagnostic_info"):
                diagnostic_info.update_ratio(ratio)
                diagnostic_info.update_slow_iter(-1)  # n/a, unless otherwise update
            if ratio < params("tr_radius.eta1"):  # ratio < 0.1
                control.delta = min(params("tr_radius.gamma_dec") * control.delta, dnorm)
                if params("logging.save_diagnostic_info"):
                    # logging.info("Last eval was for unsuccessful step (ratio = %g)" % ratio)
                    diagnostic_info.update_iter_type(ITER_ACCEPTABLE_NO_GEOM if ratio > 0.0
                                                     else ITER_UNSUCCESSFUL_NO_GEOM)  # we flag geom update below
            elif ratio <= params("tr_radius.eta2"):  # 0.1 <= ratio <= 0.7
                control.delta = max(params("tr_radius.gamma_dec") * control.delta, dnorm)
                if params("logging.save_diagnostic_info"):
                    # logging.info("Last eval was for acceptable step (ratio = %g)" % ratio)
                    diagnostic_info.update_iter_type(ITER_SUCCESSFUL)
            else:  # (ratio > eta2 = 0.7)
                control.delta = min(max(params("tr_radius.gamma_inc") * control.delta,
                                        params("tr_radius.gamma_inc_overline") * dnorm), 1.0e10)
                if params("logging.save_diagnostic_info"):
                    # logging.info("Last eval was for successful step (ratio = %g)" % ratio)
                    diagnostic_info.update_iter_type(ITER_VERY_SUCCESSFUL)
            if control.delta <= 1.5 * control.rho:  # cap trust region radius at rho
                control.delta = control.rho

            # Steps for successful steps
            if ratio > 0.0:
                # Re-select knew, allowing knew=kopt this time
                knew, exit_info = control.choose_point_to_replace(d, skip_kopt=False)
                if exit_info is not None:
                    if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                            "restarts.use_soft_restarts"):
                        number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                        exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                         x_in_abs_coords_to_save=None, f_to_save=None,
                                                         nsamples_to_save=None)
                        if exit_info is not None:
                            nruns_so_far += 1
                            break  # quit
                        current_iter = -1
                        nruns_so_far += 1
                        rhoend = params("restarts.rhoend_scale") * rhoend
                        restart_auto_detect_full = False
                        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        continue  # next iteration
                    else:
                        nruns_so_far += 1
                        break  # quit

            # Update point
            logging.debug("Updating with knew = %i" % knew)
            control.model.change_point(knew, xnew, f_list[0])  # expect step, not absolute x
            for i in range(1, num_samples_run):
                control.model.add_new_sample(knew, f_extra=f_list[i])

            # Termination check: slow iterations [needs to be after updated with new point, as use model.fopt()
            if ratio > 0.0:
                this_iter_slow, should_terminate = control.terminate_from_slow_iterations(current_iter, params)
                if params("logging.save_diagnostic_info"):
                    diagnostic_info.update_slow_iter(1 if this_iter_slow else 0)
                if should_terminate:
                    logging.info("Slow iteration  - terminating/restarting")
                    if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                        number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                        exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                         x_in_abs_coords_to_save=None, f_to_save=None,
                                                         nsamples_to_save=None)
                        if exit_info is not None:
                            nruns_so_far += 1
                            break  # quit
                        current_iter = -1
                        nruns_so_far += 1
                        rhoend = params("restarts.rhoend_scale") * rhoend
                        restart_auto_detect_full = False
                        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        continue  # next iteration
                    else:
                        exit_info = ExitInformation(EXIT_SLOW_WARNING, "Maximum slow iterations reached")
                        nruns_so_far += 1
                        break  # quit

                # Update list of successful steps
                this_step_was_not_improvement = control.model.fsave is not None and control.model.fopt() > control.model.fsave
                succ_steps_not_improvement.pop()  # remove last item
                succ_steps_not_improvement.insert(0, this_step_was_not_improvement)  # add at beginning
                # Terminate (not restart) if all are True
                if all(succ_steps_not_improvement):
                    exit_info = ExitInformation(EXIT_FALSE_SUCCESS_WARNING, "Maximum false successful steps reached")
                    nruns_so_far += 1
                    break  # quit

            if ratio >= params("tr_radius.eta1"):  # ratio >= 0.1
                continue  # next iteration

            # Auto-detection of restarts - check if we should do a restart
            if params("restarts.use_restarts") and params("restarts.auto_detect") and restart_auto_detect_full:
                do_restart = False
                iters_delta_flat = np.where(np.abs(restart_auto_detect_delta[1:]-restart_auto_detect_delta[:-1])<1e-15)[0]
                iters_delta_down = np.where(restart_auto_detect_delta[1:] - restart_auto_detect_delta[:-1] < -1e-15)[0]
                iters_delta_up = np.where(restart_auto_detect_delta[1:] - restart_auto_detect_delta[:-1] > 1e-15)[0]
                if len(iters_delta_up) == 0 and len(iters_delta_down) > 2*len(iters_delta_flat):
                    # no very successful iterations, and twice as many unsuccessful than moderately successful iterations

                    # If delta criteria met, check chg_grad and chg_hess criteria
                    # Fit line to k vs. log(||chg_grad||_2) and log(||chg_hess||_F) separately; both have to increase
                    slope, intercept, r_value, p_value, std_err = STAT.linregress(np.arange(len(restart_auto_detect_chg_grad)),
                                                                                  np.log(restart_auto_detect_chg_grad))
                    if control.model.npt() > control.n() + 1:
                        slope2, intercept2, r_value2, p_value2, std_err2 = STAT.linregress(np.arange(len(restart_auto_detect_chg_hess)),
                                                                                  np.log(restart_auto_detect_chg_hess))
                    else:
                        slope2, intercept2, r_value2, p_value2, std_err2 = slope, intercept, r_value, p_value, std_err

                    logging.debug("Iter %g: (slope, intercept, r_value) = (%g, %g, %g)" % (current_iter, slope, intercept, r_value))
                    if min(slope, slope2) > params("restarts.auto_detect.min_chg_model_slope") \
                            and min(r_value, r_value2) > params("restarts.auto_detect.min_correl"):
                        # increasing trend, with at least some positive correlation
                        do_restart = True
                    else:
                        do_restart = False

                if do_restart and params("restarts.use_soft_restarts"):
                    logging.info("Auto detection: need to do a restart")
                    logging.debug("delta history = %s" % str(restart_auto_detect_delta))
                    logging.debug("chg_grad history = %s" % str(restart_auto_detect_chg_grad))
                    logging.debug("chg_hess history = %s" % str(restart_auto_detect_chg_hess))
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, f_to_save=None,
                                                     nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                elif do_restart:
                    logging.info("Auto detection: need to do a restart")
                    exit_info = ExitInformation(EXIT_AUTO_DETECT_RESTART_WARNING, "Auto-detected restart")
                    nruns_so_far += 1
                    break  # quit
                    # If not doing restart, just continue as below (geom steps, etc.)

            # Otherwise (ratio < eta1 = 0.1), check & fix geometry
            logging.debug("Checking and possibly improving geometry (unsuccessful step)")
            distsq = max((2.0 * control.delta) ** 2, (10.0 * control.rho) ** 2)
            update_delta = False
            number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
            did_fix_geom, exit_info = control.check_and_fix_geometry(distsq, update_delta, number_of_samples, params)
            if dnorm > control.rho:
                control.last_successful_iter = current_iter

            if exit_info is not None:
                if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                        "restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, f_to_save=None,
                                                     nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    nruns_so_far += 1
                    break  # quit

            if did_fix_geom:
                if params("logging.save_diagnostic_info"):
                    diagnostic_info.update_iter_type(ITER_ACCEPTABLE_GEOM if ratio > 0.0 else ITER_UNSUCCESSFUL_GEOM)
                continue  # next iteration

            # If we didn't fix geometry but we still got an objective reduction (i.e. 0 < ratio < eta1 = 0.1), continue
            if ratio > 0.0:
                continue  # next iteration

            # Otherwise, ratio <= 0 (i.e. delta was reduced) and we didn't fix geometry - check if we need to reduce rho
            if max(control.delta, dnorm) > control.rho:
                continue  # next iteration
            elif control.rho > rhoend:
                # Reduce rho
                control.reduce_rho(current_iter, params)
                logging.info("New rho = %g after %i function evaluations" % (control.rho, control.nf))
                if control.n() < params("logging.n_to_print_whole_x_vector"):
                    logging.debug("Best so far: f = %.15g at x = " % (control.model.fopt())
                                  + str(control.model.xopt(abs_coordinates=True)))
                else:
                    logging.debug("Best so far: f = %.15g at x = [...]" % (control.model.fopt()))
                continue  # next iteration
            else:
                # Quit on rho=rhoend
                if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, f_to_save=None, nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_grad = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chg_hess = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    exit_info = ExitInformation(EXIT_SUCCESS, "rho has reached rhoend")
                    nruns_so_far += 1
                    break  # quit
                    # (end trust region step)
    # (end main loop)

    # Quit & return the important information
    x, f, gradmin, hessmin, nsamples = control.model.get_final_results()
    logging.debug("At return from solver, number of function evals = %i" % nf)
    logging.debug("Smallest objective value = %.15g at x = " % f + str(x))
    return x, f, gradmin, hessmin, nsamples, control.nf, control.nx, nruns_so_far, exit_info, diagnostic_info


def solve(objfun, x0, args=(), bounds=None, npt=None, rhobeg=None, rhoend=1e-8, maxfun=None, nsamples=None, user_params=None,
          objfun_has_noise=False, seek_global_minimum=False, scaling_within_bounds=False):
    n = len(x0)
    if type(x0) == list:
        x0 = np.array(x0, dtype=np.float)

    # Set missing inputs (if not specified) to some sensible defaults
    if bounds is None:
        xl = None
        xu = None
        scaling_within_bounds = False
    else:
        assert len(bounds) == 2, "bounds must be a 2-tuple of (lower, upper), where both are arrays of size(x0)"
        xl = bounds[0]
        if type(xl) == list:
            xl = np.array(xl, dtype=np.float)
        xu = bounds[1]
        if type(xu) == list:
            xu = np.array(xu, dtype=np.float)
    
    exit_info = None
    if seek_global_minimum and (xl is None or xu is None):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "If seeking global minimum, must specify upper and lower bounds")

    if xl is None:
        xl = -1e20 * np.ones((n,))  # unconstrained
    if xu is None:
        xu = 1e20 * np.ones((n,))  # unconstrained
    if npt is None:
        npt = (n + 1) * (n + 2) // 2 if objfun_has_noise else 2 * n + 1
    if rhobeg is None:
        rhobeg = 0.1 if scaling_within_bounds else 0.1 * max(np.max(np.abs(x0)), 1.0)
    if maxfun is None:
        maxfun = min(100 * (n + 1), 1000)  # 100 gradients, capped at 1000
    else:
        maxfun = int(maxfun)
    if nsamples is None:
        nsamples = lambda delta, rho, iter, nruns: 1  # no averaging

    # Set parameters
    params = ParameterList(int(n), int(npt), int(maxfun), objfun_has_noise=objfun_has_noise, seek_global_minimum=seek_global_minimum)
    if user_params is not None:
        for (key, val) in user_params.items():
            params(key, new_value=val)

    scaling_changes = None
    if scaling_within_bounds:
        shift = xl.copy()
        scale = xu - xl
        scaling_changes = (shift, scale)

    x0 = apply_scaling(x0, scaling_changes)
    xl = apply_scaling(xl, scaling_changes)
    xu = apply_scaling(xu, scaling_changes)

    # Input & parameter checks
    if exit_info is None and npt < n + 1:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "npt must be >= n+1")

    if exit_info is None and npt > (n + 1) * (n + 2) // 2:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "npt must be <= (n+1)*(n+2)/2")

    if exit_info is None and rhobeg < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhobeg must be strictly positive")

    if exit_info is None and rhoend < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhoend must be strictly positive")

    if exit_info is None and rhobeg <= rhoend:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhobeg must be > rhoend")

    if exit_info is None and maxfun <= 0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "maxfun must be strictly positive")

    if exit_info is None and np.shape(x0) != (n,):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "x0 must be a vector")

    if exit_info is None and np.shape(x0) != np.shape(xl):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "lower bounds must have same shape as x0")

    if exit_info is None and np.shape(x0) != np.shape(xu):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "upper bounds must have same shape as x0")

    if exit_info is None and np.min(xu - xl) < 2.0 * rhobeg:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "gap between lower and upper must be at least 2*rhobeg")

    if maxfun <= npt:
        warnings.warn("maxfun <= npt: Are you sure your budget is large enough?", RuntimeWarning)

    # Check invalid parameter values

    all_ok, bad_keys = params.check_all_params(npt)
    if exit_info is None and not all_ok:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "Bad parameters: %s" % str(bad_keys))

    # Ensure no doubling-up on noise estimates
    if exit_info is None and params("noise.quit_on_noise_level"):
        if params("noise.multiplicative_noise_level") is None:
            if params("noise.additive_noise_level") is None:
                params("noise.additive_noise_level", new_value=0.0)  # do not quit on noise level
        else:
            if params("noise.additive_noise_level") is not None:
                exit_info = ExitInformation(EXIT_INPUT_ERROR,
                                            "Must have exactly one of additive or multiplicative noise estimate")

    if exit_info is None and params("init.run_in_parallel") and not params("init.random_initial_directions"):
        exit_info = ExitInformation(EXIT_INPUT_ERROR,
                                    "Parallel initialisation not yet developed for coordinate initial directions")

    # If we had an input error, quit gracefully
    if exit_info is not None:
        exit_flag = exit_info.flag
        exit_msg = exit_info.message(with_stem=True)
        results = OptimResults(None, None, None, None, 0, 0, 0, exit_flag, exit_msg)
        return results

    # Enforce lower & upper bounds on x0
    idx = (xl < x0) & (x0 <= xl + rhobeg)
    if np.any(idx):
        warnings.warn("x0 too close to lower bound, adjusting", RuntimeWarning)
    x0[idx] = xl[idx] + rhobeg

    idx = (x0 <= xl)
    if np.any(idx):
        warnings.warn("x0 below lower bound, adjusting", RuntimeWarning)
    x0[idx] = xl[idx]

    idx = (xu - rhobeg <= x0) & (x0 < xu)
    if np.any(idx):
        warnings.warn("x0 too close to upper bound, adjusting", RuntimeWarning)
    x0[idx] = xu[idx] - rhobeg

    idx = (x0 >= xu)
    if np.any(idx):
        warnings.warn("x0 above upper bound, adjusting", RuntimeWarning)
    x0[idx] = xu[idx]

    # Call main solver (first time)
    diagnostic_info = DiagnosticInfo()
    nruns = 0
    nf = 0
    nx = 0
    xmin, fmin, gradmin, hessmin, nsamples_min, nf, nx, nruns, exit_info, diagnostic_info = \
        solve_main(objfun, x0, args, xl, xu, npt, rhobeg, rhoend, maxfun, nruns, nf, nx, nsamples, params,
                    diagnostic_info, scaling_changes)

    # Hard restarts loop
    last_successful_run = nruns
    total_unsuccessful_restarts = 0
    reduction_last_run = True
    _rhobeg = rhobeg
    _rhoend = rhoend
    while params("restarts.use_restarts") and not params("restarts.use_soft_restarts") and nf < maxfun and \
            exit_info.able_to_do_restart() and nruns - last_successful_run < params("restarts.max_unsuccessful_restarts")\
            and total_unsuccessful_restarts < params("restarts.max_unsuccessful_restarts_total"):
        _rhoend = params("restarts.rhoend_scale") * _rhoend
        
        if not reduction_last_run:
            _rhobeg = _rhobeg * params("restarts.rhobeg_scale_after_unsuccessful_restart")
        
        logging.info("Restarting from finish point (f = %g) after %g function evals; using rhobeg = %g and rhoend = %g"
                     % (fmin, nf, _rhobeg, _rhoend))
        if params("restarts.hard.use_old_fk"):
            xmin2, fmin2, gradmin2, hessmin2, nsamples2, nf, nx, nruns, exit_info, diagnostic_info = \
                solve_main(objfun, xmin, args, xl, xu, npt, _rhobeg, _rhoend, maxfun, nruns, nf, nx, nsamples, params,
                            diagnostic_info, scaling_changes, f0_avg_old=fmin, f0_nsamples_old=nsamples_min)
        else:
            xmin2, fmin2, gradmin2, hessmin2, nsamples2, nf, nx, nruns, exit_info, diagnostic_info = \
                solve_main(objfun, xmin, args, xl, xu, npt, _rhobeg, _rhoend, maxfun, nruns, nf, nx, nsamples, params,
                           diagnostic_info, scaling_changes)

        if fmin2 < fmin or np.isnan(fmin):
            logging.info("Successful run with new f = %s compared to old f = %s" % (fmin2, fmin))
            last_successful_run = nruns
            (xmin, fmin, nsamples_min) = (xmin2, fmin2, nsamples2)
            if gradmin2 is not None:  # may be None if finished during setup phase, in which case just use old gradient
                gradmin = gradmin2
            if hessmin2 is not None:  # may be None if finished during setup phase, in which case just use old Hessian
                hessmin = hessmin2
            reduction_last_run = True
        else:
            logging.info("Unsuccessful run with new f = %s compared to old f = %s" % (fmin2, fmin))
            reduction_last_run = False
            total_unsuccessful_restarts += 1

    if nruns - last_successful_run >= params("restarts.max_unsuccessful_restarts"):
        exit_info = ExitInformation(EXIT_SUCCESS, "Reached maximum number of consecutive unsuccessful restarts")
    elif total_unsuccessful_restarts >= params("restarts.max_unsuccessful_restarts_total"):
        exit_info = ExitInformation(EXIT_SUCCESS, "Reached maximum total number of unsuccessful restarts")

    # Process final return values & package up
    exit_flag = exit_info.flag
    exit_msg = exit_info.message(with_stem=True)
    # Un-scale gradient and Hessian
    if scaling_changes is not None:
        if gradmin is not None:
            gradmin = gradmin / scaling_changes[1]
        if hessmin is not None:
            hessmin = hessmin / np.outer(scaling_changes[1], scaling_changes[1])
    results = OptimResults(remove_scaling(xmin, scaling_changes), fmin, gradmin, hessmin, nf, nx, nruns, exit_flag, exit_msg)
    if params("logging.save_diagnostic_info"):
        df = diagnostic_info.to_dataframe(with_xk=params("logging.save_xk"))
        results.diagnostic_info = df

    logging.info("Did a total of %g run(s)" % nruns)

    return results

