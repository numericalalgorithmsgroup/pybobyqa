"""
Parameters
====

A container class for all the solver parameter values.


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

__all__ = ['ParameterList']


class ParameterList(object):
    def __init__(self, n, npt, maxfun, objfun_has_noise=False, seek_global_minimum=False):
        self.params = {}
        # Rounding error constant (for shifting base)
        self.params["general.rounding_error_constant"] = 0.1  # 0.1 in DFBOLS, 1e-3 in BOBYQA
        self.params["general.safety_step_thresh"] = 0.5  # safety step called if ||d|| <= thresh * rho
        self.params["general.check_objfun_for_overflow"] = True
        # Initialisation
        self.params["init.random_initial_directions"] = True
        self.params["init.run_in_parallel"] = False  # only available for random directions at the moment
        self.params["init.random_directions_make_orthogonal"] = True  # although random > orthogonal, avoid for init
        # Interpolation
        # underdetermined quadratic interp uses minimum norm Hessian or minimum norm (change Hessian)?
        self.params["interpolation.minimum_change_hessian"] = True
        self.params["interpolation.precondition"] = True
        # Logging
        self.params["logging.n_to_print_whole_x_vector"] = 6
        self.params["logging.save_diagnostic_info"] = False
        self.params["logging.save_poisedness"] = True
        self.params["logging.save_xk"] = False
        # Trust Region Radius management
        self.params["tr_radius.eta1"] = 0.1
        self.params["tr_radius.eta2"] = 0.7
        self.params["tr_radius.gamma_dec"] = 0.98 if objfun_has_noise else 0.5
        self.params["tr_radius.gamma_inc"] = 2.0
        self.params["tr_radius.gamma_inc_overline"] = 4.0
        self.params["tr_radius.alpha1"] = 0.9 if objfun_has_noise else 0.1
        self.params["tr_radius.alpha2"] = 0.95 if objfun_has_noise else 0.5
        # Objective threshold
        self.params["model.abs_tol"] = -1e20
        # Slow progress thresholds
        self.params["slow.history_for_slow"] = 5
        self.params["slow.thresh_for_slow"] = 1e-8
        self.params["slow.max_slow_iters"] = 20 * n
        # Noise-based quitting
        self.params["noise.quit_on_noise_level"] = True if objfun_has_noise else False
        self.params["noise.scale_factor_for_quit"] = 1.0
        self.params["noise.multiplicative_noise_level"] = None
        self.params["noise.additive_noise_level"] = None
        # Restarts
        self.params["restarts.use_restarts"] = True if (objfun_has_noise or seek_global_minimum) else False
        self.params["restarts.max_unsuccessful_restarts"] = 10
        self.params["restarts.max_unsuccessful_restarts_total"] = 20 if seek_global_minimum else maxfun  # i.e. not used (in line with previous behaviour)
        self.params["restarts.rhobeg_scale_after_unsuccessful_restart"] = 1.1 if seek_global_minimum else 1.0
        self.params["restarts.rhoend_scale"] = 1.0  # how much to decrease rhoend by after each restart
        self.params["restarts.use_soft_restarts"] = True
        self.params["restarts.soft.num_geom_steps"] = 3
        self.params["restarts.soft.move_xk"] = True
        self.params["restarts.soft.max_fake_successful_steps"] = maxfun  # number ratio>0 steps below fsave allowed
        self.params["restarts.hard.use_old_fk"] = True  # recycle f(xk) from previous run?
        self.params["restarts.auto_detect"] = True
        self.params["restarts.auto_detect.history"] = 30
        self.params["restarts.auto_detect.min_chg_model_slope"] = 1.5e-2
        self.params["restarts.auto_detect.min_correl"] = 0.1

        self.params_changed = {}
        for p in self.params:
            self.params_changed[p] = False

    def __call__(self, key, new_value=None):  # self(key) or self(key, new_value)
        if key in self.params:
            if new_value is None:
                return self.params[key]
            else:
                if self.params_changed[key]:
                    raise ValueError("Trying to update parameter '%s' for a second time" % key)
                self.params[key] = new_value
                self.params_changed[key] = True
                return self.params[key]
        else:
            raise ValueError("Unknown parameter '%s'" % key)

    def param_type(self, key, npt):
        # Use the check_* methods below, but switch based on key
        if key == "general.rounding_error_constant":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "general.safety_step_thresh":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "general.check_objfun_for_overflow":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "init.random_initial_directions":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "init.run_in_parallel":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "init.random_directions_make_orthogonal":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "interpolation.minimum_change_hessian":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "interpolation.precondition":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "logging.n_to_print_whole_x_vector":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "logging.save_diagnostic_info":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "logging.save_poisedness":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "logging.save_xk":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "tr_radius.eta1":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.eta2":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.gamma_dec":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.gamma_inc":
            type_str, nonetype_ok, lower, upper = 'float', False, 1.0, None
        elif key == "tr_radius.gamma_inc_overline":
            type_str, nonetype_ok, lower, upper = 'float', False, 1.0, None
        elif key == "tr_radius.alpha1":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.alpha2":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "model.abs_tol":
            type_str, nonetype_ok, lower, upper = 'float', False, None, None
        elif key == "slow.history_for_slow":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "slow.thresh_for_slow":
            type_str, nonetype_ok, lower, upper = 'float', False, 0, None
        elif key == "slow.max_slow_iters":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "noise.quit_on_noise_level":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "noise.scale_factor_for_quit":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "noise.multiplicative_noise_level":
            type_str, nonetype_ok, lower, upper = 'float', True, 0.0, None
        elif key == "noise.additive_noise_level":
            type_str, nonetype_ok, lower, upper = 'float', True, 0.0, None
        elif key == "restarts.use_restarts":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "restarts.max_unsuccessful_restarts":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "restarts.max_unsuccessful_restarts_total":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "restarts.rhobeg_scale_after_unsuccessful_restart":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "restarts.rhoend_scale":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "restarts.use_soft_restarts":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "restarts.soft.num_geom_steps":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "restarts.soft.move_xk":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "restarts.soft.max_fake_successful_steps":
            type_str, nonetype_ok, lower, upper = 'int', False, 1, None
        elif key == "restarts.hard.use_old_fk":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "restarts.increase_npt":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "restarts.increase_npt_amt":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "restarts.max_npt":
            type_str, nonetype_ok, lower, upper = 'int', False, npt, None
        elif key == "restarts.auto_detect":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "restarts.auto_detect.history":
            type_str, nonetype_ok, lower, upper = 'int', False, 1, None
        elif key == "restarts.auto_detect.min_chg_model_slope":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "restarts.auto_detect.min_correl":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        else:
            assert False, "ParameterList.param_type() has unknown key: %s" % key
        return type_str, nonetype_ok, lower, upper

    def check_param(self, key, value, npt):
        type_str, nonetype_ok, lower, upper = self.param_type(key, npt)
        if type_str == 'int':
            return check_integer(value, lower=lower, upper=upper, allow_nonetype=nonetype_ok)
        elif type_str == 'float':
            return check_float(value, lower=lower, upper=upper, allow_nonetype=nonetype_ok)
        elif type_str == 'bool':
            return check_bool(value, allow_nonetype=nonetype_ok)
        elif type_str == 'str':
            return check_str(value, allow_nonetype=nonetype_ok)
        else:
            assert False, "Unknown type_str '%s' for parameter '%s'" % (type_str, key)

    def check_all_params(self, npt):
        bad_keys = []
        for key in self.params:
            if not self.check_param(key, self.params[key], npt):
                bad_keys.append(key)
        return len(bad_keys) == 0, bad_keys


def check_integer(val, lower=None, upper=None, allow_nonetype=False):
    # Check that val is an integer and (optionally) that lower <= val <= upper
    if val is None:
        return allow_nonetype
    elif not isinstance(val, int):
        return False
    else:  # is integer
        return (lower is None or val >= lower) and (upper is None or val <= upper)


def check_float(val, lower=None, upper=None, allow_nonetype=False):
    # Check that val is a float and (optionally) that lower <= val <= upper
    if val is None:
        return allow_nonetype
    elif not isinstance(val, float):
        return False
    else:  # is integer
        return (lower is None or val >= lower) and (upper is None or val <= upper)


def check_bool(val, allow_nonetype=False):
    if val is None:
        return allow_nonetype
    else:
        return isinstance(val, bool)


def check_str(val, allow_nonetype=False):
    if val is None:
        return allow_nonetype
    else:
        return isinstance(val, str) or isinstance(val, unicode)
