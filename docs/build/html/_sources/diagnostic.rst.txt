Diagnostic Information
======================
In :doc:`userguide`, we saw that the output of Py-BOBYQA returns a container which includes diagnostic information about the progress of the algorithm (:code:`soln.diagnostic_info`). This object is a `Pandas <http://pandas.pydata.org/>`_ DataFrame, with one row per iteration of the algorithm. In this section, we explain the meaning of each type of output (the columns of the DataFrame).

To save this information to a CSV file, use:

  .. code-block:: python
  
      # Previously: define objfun and x0
      
      # Turn on diagnostic information
      user_params = {'logging.save_diagnostic_info': True}
      
      # Call Py-BOBYQA
      soln = pybobyqa.solve(objfun, x0, user_params=user_params)
      
      # Save diagnostic info to CSV
      soln.diagnostic_info.to_csv("myfile.csv")
 
Depending on exactly how Py-BOBYQA terminates, the last row of results may not be fully populated.
 
Current Iterate
---------------
* :code:`xk` - Best point found so far (current iterate). This is only saved if :code:`user_params['logging.save_xk'] = True`.
* :code:`fk` - The value of :math:`f` at the current iterate.

Trust Region
------------
* :code:`rho` - The lower bound on the trust region radius :math:`\rho_k`.
* :code:`delta` - The trust region radius :math:`\Delta_k`.
* :code:`norm_sk` - The norm of the trust region step :math:`\|s_k\|`.

Model Interpolation
-------------------
* :code:`npt` - The number of interpolation points.
* :code:`interpolation_error` - The sum of squares of the interpolation errors from the interpolated model.
* :code:`interpolation_condition_number` - The condition number of the matrix in the interpolation linear system.
* :code:`interpolation_change_g_norm` - The norm of the change in model gradient at this iteration, :math:`\|g_k-g_{k-1}\|`.
* :code:`interpolation_change_H_norm` - The Frobenius norm of the change in model Hessian at this iteration, :math:`\|H_k-H_{k-1}\|_F`.
* :code:`poisedness` - The smallest value of :math:`\Lambda` for which the current interpolation set :math:`Y_k` is :math:`\Lambda`-poised in the current trust region. This is the most expensive piece of information to compute, and is only computed if :code:`user_params['logging.save_poisedness' = True`.
* :code:`max_distance_xk` - The maximum distance from any interpolation point to the current iterate.
* :code:`norm_gk` - The norm of the model gradient :math:`\|g_k\|`.

Iteration Count
---------------
* :code:`nruns` - The number of times the algorithm has been restarted.
* :code:`nf` - The number of objective evaluations so far (see :code:`soln.nf`)
* :code:`nx` - The number of points at which the objective has been evaluated so far (see :code:`soln.nx`)
* :code:`nsamples` - The total number of objective evaluations used for all current interpolation points.
* :code:`iter_this_run` - The number of iterations since the last restart.
* :code:`iters_total` - The total number of iterations so far.

Algorithm Progress
------------------
* :code:`iter_type` - A text description of what type of iteration we had (e.g. Successful, Safety, etc.)
* :code:`ratio` - The ratio of actual to predicted objective reduction in the trust region step.
* :code:`slow_iter` - Equal to 1 if the current iteration is successful but slow, 0 if is successful but not slow, and -1 if was not successful.

