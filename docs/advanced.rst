Advanced Usage
==============
This section describes different optional user parameters available in Py-BOBYQA.

In the last section (:doc:`userguide`), we introduced :code:`pybobyqa.solve()`, which has the optional input :code:`user_params`. This is a Python dictionary of user parameters. We will now go through the settings which can be changed in this way. More details are available in the paper [CFMR2018]_.

The default values, used if no override is given, in some cases vary depending on whether :code:`objfun` has stochastic noise; that is, whether evaluating :code:`objfun(x)` several times at the same :code:`x` gives the same result or not. Whether or not this is the case is determined by the :code:`objfun_has_noise` input to :code:`pybobyqa.solve()` (and not by inspecting :code:`objfun`, for instance).

General Algorithm Parameters
----------------------------
* :code:`general.rounding_error_constant` - Internally, all interpolation points are stored with respect to a base point :math:`x_b`; that is, we store :math:`\{y_t-x_b\}`, which reduces the risk of roundoff errors. We shift :math:`x_b` to :math:`x_k` when :math:`\|s_k\| \leq \text{const}\|x_k-x_b\|`, where 'const' is this parameter. Default is 0.1.
* :code:`general.safety_step_thresh` - Threshold for when to call the safety step, :math:`\|s_k\| \leq \gamma_S \rho_k`. Default is :math:`\gamma_S =0.5`.
* :code:`general.check_objfun_for_overflow` - Whether to cap the value of :math:`r_i(x)` when they are large enough that an OverflowError will be encountered when trying to evaluate :math:`f(x)`. Default is :code:`True`. 

Logging and Output
------------------
* :code:`logging.n_to_print_whole_x_vector` - If printing all function evaluations to screen/log file, the maximum :code:`len(x)` for which the full vector :code:`x` should be printed also. Default is 6.
* :code:`logging.save_diagnostic_info` - Flag so save diagnostic information at each iteration. Default is :code:`False`.
* :code:`logging.save_poisedness` - If saving diagnostic information, whether to include the :math:`\Lambda`-poisedness of :math:`Y_k` in the diagnostic information. This is the most computationally expensive piece of diagnostic information. Default is :code:`True`.
* :code:`logging.save_xk` - If saving diagnostic information, whether to include the full vector :math:`x_k`. Default is :code:`False`.

Initialization of Points
------------------------
* :code:`init.random_initial_directions` - Build the initial interpolation set using random directions (as opposed to coordinate directions). Default is :code:`True`.
* :code:`init.random_directions_make_orthogonal` - If building initial interpolation set with random directions, whether or not these should be orthogonalized. Default is :code:`True`.
* :code:`init.run_in_parallel` - If using random directions, whether or not to ask for all :code:`objfun` to be evaluated at all points without any intermediate processing. Default is :code:`False`.

Trust Region Management
-----------------------
* :code:`tr_radius.eta1` - Threshold for unsuccessful trust region iteration, :math:`\eta_1`. Default is 0.1. 
* :code:`tr_radius.eta2` - Threshold for very successful trust region iteration, :math:`\eta_2`. Default is 0.7. 
* :code:`tr_radius.gamma_dec` - Ratio to decrease :math:`\Delta_k` in unsuccessful iteration, :math:`\gamma_{dec}`. Default is 0.5 for smooth problems or 0.98 for noisy problems (i.e. :code:`objfun_has_noise = True`). 
* :code:`tr_radius.gamma_inc` - Ratio to increase :math:`\Delta_k` in very successful iterations, :math:`\gamma_{inc}`. Default is 2. 
* :code:`tr_radius.gamma_inc_overline` - Ratio of :math:`\|s_k\|` to increase :math:`\Delta_k` by in very successful iterations, :math:`\overline{\gamma}_{inc}`. Default is 4. 
* :code:`tr_radius.alpha1` - Ratio to decrease :math:`\rho_k` by when it is reduced, :math:`\alpha_1`. Default is 0.1 for smooth problems or 0.9 for noisy problems (i.e. :code:`objfun_has_noise = True`). 
* :code:`tr_radius.alpha2` - Ratio of :math:`\rho_k` to decrease :math:`\Delta_k` by when :math:`\rho_k` is reduced, :math:`\alpha_2`. Default is 0.5 for smooth problems or 0.95 for noisy problems (i.e. :code:`objfun_has_noise = True`). 

Termination on Small Objective Value
------------------------------------
* :code:`model.abs_tol` - Tolerance on :math:`f(x_k)`; quit if :math:`f(x_k)` is below this value. Default is :math:`-10^{20}`. 

Termination on Slow Progress
----------------------------
* :code:`slow.history_for_slow` - History used to determine whether the current iteration is 'slow'. Default is 5. 
* :code:`slow.thresh_for_slow` - Threshold for objective decrease used to determine whether the current iteration is 'slow'. Default is :math:`10^{-8}`. 
* :code:`slow.max_slow_iters` - Number of consecutive slow successful iterations before termination (or restart). Default is :code:`20*len(x0)`. 

Stochastic Noise Information
----------------------------
* :code:`noise.quit_on_noise_level` - Flag to quit (or restart) if all :math:`f(y_t)` are within noise level of :math:`f(x_k)`. Default is :code:`False` for smooth problems or :code:`True` for noisy problems. 
* :code:`noise.scale_factor_for_quit` - Factor of noise level to use in termination criterion. Default is 1. 
* :code:`noise.multiplicative_noise_level` - Multiplicative noise level in :math:`f`. Can only specify one of multiplicative or additive noise levels. Default is :code:`None`. 
* :code:`noise.additive_noise_level` - Additive noise level in :math:`f`. Can only specify one of multiplicative or additive noise levels. Default is :code:`None`. 

Interpolation Management
--------------------------------
* :code:`interpolation.precondition` - whether or not to scale the interpolation linear system to improve conditioning. Default is :code:`True`.
* :code:`interpolation.minimum_change_hessian` - whether to solve the underdetermined quadratic interpolation problem by minimizing the Frobenius norm of the Hessian, or change in Hessian. Default is :code:`True`.

Multiple Restarts
-----------------
* :code:`restarts.use_restarts` - Whether to do restarts when :math:`\rho_k` reaches :math:`\rho_{end}`, or (optionally) when all points are within noise level of :math:`f(x_k)`. Default is :code:`False` for smooth problems or :code:`True` for noisy problems. 
* :code:`restarts.max_unsuccessful_restarts` - Maximum number of consecutive unsuccessful restarts allowed (i.e.~restarts which did not reduce the objective further). Default is 10. 
* :code:`restarts.rhoend_scale` - Factor to reduce :math:`\rho_{end}` by with each restart. Default is 1. 
* :code:`restarts.use_soft_restarts` - Whether to use soft or hard restarts. Default is :code:`True`. 
* :code:`restarts.soft.num_geom_steps` - For soft restarts, the number of points to move. Default is 3. 
* :code:`restarts.soft.move_xk` - For soft restarts, whether to preserve :math:`x_k`, or move it to the best new point evaluated. Default is :code:`True`. 
* :code:`restarts.hard.use_old_fk` - If using hard restarts, whether or not to recycle the objective value at the best iterate found when performing a restart. This saves one objective evaluation. Default is :code:`True`.
* :code:`restarts.soft.max_fake_successful_steps` - The maximum number of successful steps in a given run where the new (smaller) objective value is larger than the best value found in a previous run. Default is :code:`maxfun`, the input to :code:`pybobyqa.solve()`.
* :code:`restarts.auto_detect` - Whether or not to automatically determine when to restart. This is an extra condition, and restarts can still be triggered by small trust region radius, etc. Default is :code:`True`.
* :code:`restarts.auto_detect.history` - How many iterations of data on model changes and trust region radii to store. There are two criteria used: trust region radius decreases (no increases over the history, more decreases than no changes), and change in model Jacobian (consistently increasing trend as measured by slope and correlation coefficient of line of best fit). Default is 30.
* :code:`restarts.auto_detect.min_chg_model_slope` - Minimum rate of increase of :math:`\log(\|g_k-g_{k-1}\|)` and :math:`\log(\|H_k-H_{k-1}\|_F)` over the past iterations to cause a restart. Default is 0.015.
* :code:`restarts.auto_detect.min_correl` - Minimum correlation of the data sets :math:`(k, \log(\|g_k-g_{k-1}\|))` and :math:`(k, \log(\|H_k-H_{k-1}\|_F))` required to cause a restart. Default is 0.1.


References
----------

.. [CFMR2018]   
   C. Cartis, J. Fiala, B. Marteau and L. Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://arxiv.org/abs/1804.00154>`_, technical report, University of Oxford, (2018).

