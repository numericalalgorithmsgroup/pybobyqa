"""
Diagnostic Info
====

A class containing diagnostic information (optionally) produced by the solver.


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
import pandas as pd
from .util import remove_scaling


__all__ = ['DiagnosticInfo', 'ITER_VERY_SUCCESSFUL', 'ITER_SUCCESSFUL', 'ITER_ACCEPTABLE_GEOM',
           'ITER_ACCEPTABLE_NO_GEOM', 'ITER_UNSUCCESSFUL_GEOM', 'ITER_UNSUCCESSFUL_NO_GEOM', 'ITER_SAFETY']


ITER_VERY_SUCCESSFUL = "Very successful"   # ratio >= 0.7, no geometry update
ITER_SUCCESSFUL = "Successful"  # 0.1 <= ratio < 0.7, no geometry update
ITER_ACCEPTABLE_GEOM = "Acceptable (geom fixed)"  # 0 <= ratio < 0.1, with geometry update
ITER_ACCEPTABLE_NO_GEOM = "Acceptable (geom not fixed)"  # 0 <= ratio < 0.1, without geometry update
ITER_UNSUCCESSFUL_GEOM = "Unsuccessful (geom fixed)"  # ratio < 0, with geometry update
ITER_UNSUCCESSFUL_NO_GEOM = "Unsuccessful (geom not fixed)"  # ratio < 0, without geometry update (possibly rho reduced)
ITER_SAFETY = "Safety"  # safety step taken (||s|| too small compared to rho)


class DiagnosticInfo(object):
    def __init__(self):
        self.data = {}
        # Initialise everything we want to store
        self.data["xk"] = []
        self.data["fk"] = []

        self.data["rho"] = []
        self.data["delta"] = []

        self.data["interpolation_error"] = []
        self.data["interpolation_condition_number"] = []
        self.data["interpolation_change_g_norm"] = []
        self.data["interpolation_change_H_norm"] = []
        self.data["poisedness"] = []
        self.data["max_distance_xk"] = []
        self.data["norm_gk"] = []
        self.data["norm_sk"] = []

        self.data["nruns"] = []
        self.data["nf"] = []
        self.data["nx"] = []
        self.data["npt"] = []
        self.data["nsamples"] = []
        self.data["iter_this_run"] = []
        self.data["iters_total"] = []

        self.data["iter_type"] = []
        self.data["ratio"] = []
        self.data["slow_iter"] = []
        return

    def to_dataframe(self, with_xk=False):
        data_to_save = {}
        for key in self.data:
            if key == "xk" and not with_xk:
                continue  # skip
            data_to_save[key] = self.data[key]
        return pd.DataFrame(data_to_save)

    def to_csv(self, filename):
        df = self.to_dataframe()
        df.to_csv(filename)

    def save_info_from_control(self, control, nruns, iter_this_run, save_poisedness=True):
        self.data["iters_total"].append(len(self.data["iters_total"]))
        self.data["nruns"].append(nruns)
        self.data["iter_this_run"].append(iter_this_run)
        # Get what info we can out of a control
        self.data["nf"].append(control.nf)
        self.data["nx"].append(control.nx)
        self.data["delta"].append(control.delta)
        self.data["rho"].append(control.rho)
        # And from a model?
        self.data["npt"].append(control.model.npt())
        x, f, gradmin, hessmin, nsamples = control.model.get_final_results()
        self.data["xk"].append(remove_scaling(x, control.scaling_changes))
        self.data["fk"].append(f)
        self.data["nsamples"].append(np.sum(control.model.nsamples))
        self.data["max_distance_xk"].append(np.sqrt(np.max(control.model.distances_to_xopt())))
        # Poisedness is expensive to compute (unlike everything else here), so allow the option to not calculate
        if save_poisedness:
            self.data["poisedness"].append(control.model.poisedness_constant(control.delta))
        else:
            self.data["poisedness"].append(0.0)
        # The other things we can't get just yet, so save a default value there for now
        self.data["interpolation_error"].append(None)
        self.data["interpolation_condition_number"].append(None)
        self.data["interpolation_change_g_norm"].append(None)
        self.data["interpolation_change_H_norm"].append(None)
        self.data["norm_gk"].append(None)
        self.data["norm_sk"].append(None)
        self.data["iter_type"].append(None)
        self.data["ratio"].append(None)
        self.data["slow_iter"].append(None)
        return

    def update_interpolation_information(self, interp_error, interp_cond_num, norm_change_g,
                                         norm_change_H, norm_gk, norm_sk):
        self.data["interpolation_error"][-1] = interp_error
        self.data["interpolation_condition_number"][-1] = interp_cond_num
        self.data["interpolation_change_g_norm"][-1] = norm_change_g
        self.data["interpolation_change_H_norm"][-1] = norm_change_H
        self.data["norm_gk"][-1] = norm_gk
        self.data["norm_sk"][-1] = norm_sk
        return

    def update_ratio(self, ratio):
        self.data["ratio"][-1] = ratio
        return

    def update_iter_type(self, iter_type):
        self.data["iter_type"][-1] = iter_type
        return

    def update_slow_iter(self, slow_iter):
        self.data["slow_iter"][-1] = slow_iter
        return
