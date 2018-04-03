Using Py-BOBYQA
===============
This section describes the main interface to Py-BOBYQA and how to use it.

Nonlinear Minimization
----------------------
Py-BOBYQA is designed to solve the local optimization problem

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x) \\
   \text{s.t.} &\quad  a \leq x \leq b

where the bound constraints :math:`a \leq x \leq b` are optional. The objective function :math:`f(x)` is usually nonlinear and nonquadratic. If you know your objective is linear or quadratic, you should consider a solver designed for such functions (see `here <https://neos-guide.org/Optimization-Guide>`_ for details).

Py-BOBYQA iteratively constructs an interpolation-based model for the objective, and determines a step using a trust-region framework.
For an in-depth technical description of the algorithm see the paper [CFMR2018]_.

How to use Py-BOBYQA
--------------------
The main interface to Py-BOBYQA is via the function :code:`solve`

  .. code-block:: python
  
      soln = pybobyqa.solve(objfun, x0)

The input :code:`objfun` is a Python function which takes an input :math:`x\in\mathbb{R}^n` and returns the objective value :math:`f(x)\in\mathbb{R}`. The input of :code:`objfun` must be one-dimensional NumPy arrays (i.e. with :code:`x.shape == (n,)`) and the output must be a single Float.

The input :code:`x0` is the starting point for the solver, and (where possible) should be set to be the best available estimate of the true solution :math:`x_{min}\in\mathbb{R}^n`. It should be specified as a one-dimensional NumPy array (i.e. with :code:`x0.shape == (n,)`).
As Py-BOBYQA is a local solver, providing different values for :code:`x0` may cause it to return different solutions, with possibly different objective values.

The output of :code:`pybobyqa.solve` is an object containing:

* :code:`soln.x` - an estimate of the solution, :math:`x_{min}\in\mathbb{R}^n`, a one-dimensional NumPy array.
* :code:`soln.f` - the objective value at the calculated solution, :math:`f(x_{min})`, a Float.
* :code:`soln.gradient` - an estimate of the gradient vector of first derivatives of the objective, :math:`g_i \approx \partial f(x_{min})/\partial x_i`, a NumPy array of length :math:`n`.
* :code:`soln.hessian` - an estimate of the Hessian matrix of second derivatives of the objective, :math:`H_{i,j} \approx \partial^2 f(x_{min})/\partial x_i \partial x_j`, a NumPy array of size :math:`n\times n`.
* :code:`soln.nf` - the number of evaluations of :code:`objfun` that the algorithm needed, an Integer.
* :code:`soln.nx` - the number of points :math:`x` at which :code:`objfun` was evaluated, an Integer. This may be different to :code:`soln.nf` if sample averaging is used.
* :code:`soln.nruns` - the number of runs performed by Py-BOBYQA (more than 1 if using multiple restarts), an Integer.
* :code:`soln.flag` - an exit flag, which can take one of several values (listed below), an Integer.
* :code:`soln.msg` - a description of why the algorithm finished, a String.
* :code:`soln.diagnostic_info` - a table of diagnostic information showing the progress of the solver, a Pandas DataFrame.

The possible values of :code:`soln.flag` are defined by the following variables:

* :code:`soln.EXIT_SUCCESS` - Py-BOBYQA terminated successfully (the objective value or trust region radius are sufficiently small).
* :code:`soln.EXIT_MAXFUN_WARNING` - maximum allowed objective evaluations reached. This is the most likely return value when using multiple restarts.
* :code:`soln.EXIT_SLOW_WARNING` - maximum number of slow iterations reached.
* :code:`soln.EXIT_FALSE_SUCCESS_WARNING` - Py-BOBYQA reached the maximum number of restarts which decreased the objective, but to a worse value than was found in a previous run.
* :code:`soln.EXIT_INPUT_ERROR` - error in the inputs.
* :code:`soln.EXIT_TR_INCREASE_ERROR` - error occurred when solving the trust region subproblem.
* :code:`soln.EXIT_LINALG_ERROR` - linear algebra error, e.g. the interpolation points produced a singular linear system.

These variables are defined in the :code:`soln` object, so can be accessed with, for example

  .. code-block:: python
  
      if soln.flag == soln.EXIT_SUCCESS:
          print("Success!")

Optional Arguments
------------------
The :code:`solve` function has several optional arguments which the user may provide:

  .. code-block:: python
  
      pybobyqa.solve(objfun, x0, args=(), bounds=None, npt=None,
		  rhobeg=None, rhoend=1e-8, maxfun=None, nsamples=None, 
                  user_params=None, objfun_has_noise=False, 
                  scaling_within_bounds=False)

These arguments are:

* :code:`args` - a tuple of extra arguments passed to the objective function. This feature is new, and not yet avaiable in the PyPI version of Py-BOBYQA; instead, use Python's built-in function :code:`lambda`.
* :code:`bounds` - a tuple :code:`(lower, upper)` with the vectors :math:`a` and :math:`b` of lower and upper bounds on :math:`x` (default is :math:`a_i=-10^{20}` and :math:`b_i=10^{20}`). To set bounds for either :code:`lower` or :code:`upper`, but not both, pass a tuple :code:`(lower, None)` or :code:`(None, upper)`.
* :code:`npt` - the number of interpolation points to use (default is :code:`2*len(x0)+1`). Py-BOBYQA requires :code:`n+1 <= npt <= (n+1)*(n+2)/2` for a problem with :code:`len(x0)=n`. Larger values are particularly useful for noisy problems.
* :code:`rhobeg` - the initial value of the trust region radius (default is :math:`0.1\max(\|x_0\|_{\infty}, 1)`).
* :code:`rhoend` - minimum allowed value of trust region radius, which determines when a successful termination occurs (default is :math:`10^{-8}`).
* :code:`maxfun` - the maximum number of objective evaluations the algorithm may request (default is :math:`\min(100(n+1),1000)`).
* :code:`nsamples` - a Python function :code:`nsamples(delta, rho, iter, nrestarts)` which returns the number of times to evaluate :code:`objfun` at a given point. This is only applicable for objectives with stochastic noise, when averaging multiple evaluations at the same point produces a more accurate value. The input parameters are the trust region radius (:code:`delta`), the lower bound on the trust region radius (:code:`rho`), how many iterations the algorithm has been running for (:code:`iter`), and how many restarts have been performed (:code:`nrestarts`). Default is no averaging (i.e. :code:`nsamples(delta, rho, iter, nrestarts)=1`).
* :code:`user_params` - a Python dictionary :code:`{'param1': val1, 'param2':val2, ...}` of optional parameters. A full list of available options is given in the next section :doc:`advanced`.
* :code:`objfun_has_noise` - a flag to indicate whether or not :code:`objfun` has stochastic noise; i.e. will calling :code:`objfun(x)` multiple times at the same value of :code:`x` give different results? This is used to set some sensible default parameters (including using multiple restarts), all of which can be overridden by the values provided in :code:`user_params`.
* :code:`scaling_within_bounds` - a flag to indicate whether the algorithm should internally shift and scale the entries of :code:`x` so that the bounds become :math:`0 \leq x \leq 1`. This is useful is you are setting :code:`bounds` and the bounds have different orders of magnitude. If :code:`scaling_within_bounds=True`, the values of :code:`rhobeg` and :code:`rhoend` apply to the *shifted* variables.

In general when using optimization software, it is good practice to scale your variables so that moving each by a given amount has approximately the same impact on the objective function.
The :code:`scaling_within_bounds` flag is designed to provide an easy way to achieve this, if you have set the bounds :code:`lower` and :code:`upper`.

A Simple Example
----------------
Suppose we wish to minimize the `Rosenbrock test function <https://en.wikipedia.org/wiki/Rosenbrock_function>`_:

.. math::

   \min_{(x_1,x_2)\in\mathbb{R}^2}  &\quad  100(x_2-x_1^2)^2 + (1-x_1)^2 \\

This function has exactly one local minimum :math:`f(x_{min})=0` at :math:`x_{min}=(1,1)`. A commonly-used starting point for testing purposes is :math:`x_0=(-1.2,1)`. The following script shows how to solve this problem using Py-BOBYQA:

  .. code-block:: python
  
      # Py-BOBYQA example: minimize the Rosenbrock function
      from __future__ import print_function
      import numpy as np
      import pybobyqa

      # Define the objective function
      def rosenbrock(x):
          return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
      
      # Define the starting point
      x0 = np.array([-1.2, 1.0])
      
      # Set random seed (for reproducibility)
      np.random.seed(0)
      
      # Call Py-BOBYQA
      soln = pybobyqa.solve(rosenbrock, x0)
      
      # Display output
      print(soln)
      
Note that Py-BOBYQA is a randomized algorithm: in its first phase, it builds an internal approximation to the objective function by sampling it along random directions. In the code above, we set NumPy's random seed for reproducibility over multiple runs, but this is not required. The output of this script, showing that Py-BOBYQA finds the correct solution, is

  .. code-block:: none
  
      ****** Py-BOBYQA Results ******
      Solution xmin = [ 1.  1.]
      Objective value f(xmin) = 2.964036794e-19
      Needed 213 objective evaluations (at 213 points)
      Approximate gradient = [ -2.57280154e-08   1.26855793e-08]
      Approximate Hessian = [[ 802.90904563 -400.46022134]
       [-400.46022134  200.23335154]]
      Exit flag = 0
      Success: rho has reached rhoend
      ******************************

This and all following problems can be found in the `examples <https://github.com/numericalalgorithmsgroup/pybobyqa/tree/master/examples>`_ directory on the Py-BOBYQA Github page.

Adding Bounds and More Output
-----------------------------
We can extend the above script to add constraints. To do this, we can add the lines

  .. code-block:: python
  
      # Define bound constraints (lower <= x <= upper)
      lower = np.array([-10.0, -10.0])
      upper = np.array([0.9, 0.85])
      
      # Call Py-BOBYQA (with bounds)
      soln = pybobyqa.solve(rosenbrock, x0, bounds=(lower,upper))

Py-BOBYQA correctly finds the solution to the constrained problem:

  .. code-block:: none
  
      ****** Py-BOBYQA Results ******
      Solution xmin = [ 0.9   0.81]
      Objective value f(xmin) = 0.01
      Needed 134 objective evaluations (at 134 points)
      Approximate gradient = [ -1.99999226e-01  -4.31078784e-07]
      Approximate Hessian = [[ 649.6790222  -360.18361979]
       [-360.18361979  200.00205196]]
      Exit flag = 0
      Success: rho has reached rhoend
      ******************************

However, we also get a warning that our starting point was outside of the bounds:

  .. code-block:: none
  
      RuntimeWarning: x0 above upper bound, adjusting

Py-BOBYQA automatically fixes this, and moves :math:`x_0` to a point within the bounds, in this case :math:`x_0=(-1.2,0.85)`.

We can also get Py-BOBYQA to print out more detailed information about its progress using the `logging <https://docs.python.org/3/library/logging.html>`_ module. To do this, we need to add the following lines:

  .. code-block:: python
  
      import logging
      logging.basicConfig(level=logging.INFO, format='%(message)s')
      
      # ... (call pybobyqa.solve)

And we can now see each evaluation of :code:`objfun`:

  .. code-block:: none
  
      Function eval 1 at point 1 has f = 39.65 at x = [-1.2   0.85]
      Initialising (random directions)
      Function eval 2 at point 2 has f = 14.337296 at x = [-1.08  0.85]
      Function eval 3 at point 3 has f = 55.25 at x = [-1.2   0.73]
      ...
      Function eval 133 at point 133 has f = 0.0100000000000165 at x = [ 0.9         0.81000001]
      Function eval 134 at point 134 has f = 0.00999999999999997 at x = [ 0.9   0.81]
      Did a total of 1 run(s)

If we wanted to save this output to a file, we could replace the above call to :code:`logging.basicConfig()` with

  .. code-block:: python
  
      logging.basicConfig(filename="myfile.log", level=logging.INFO, 
                          format='%(message)s', filemode='w')

Example: Noisy Objective Evaluation
-----------------------------------
As described in :doc:`info`, derivative-free algorithms such as Py-BOBYQA are particularly useful when :code:`objfun` has noise. Let's modify the previous example to include random noise in our objective evaluation, and compare it to a derivative-based solver:

  .. code-block:: python
  
      # Py-BOBYQA example: minimize the noisy Rosenbrock function
      from __future__ import print_function
      import numpy as np
      import pybobyqa
      
      # Define the objective function
      def rosenbrock(x):
          return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
      
      # Modified objective function: add 1% Gaussian noise
      def rosenbrock_noisy(x):
          return rosenbrock(x) * (1.0 + 1e-2 * np.random.normal(size=(1,))[0])
      
      # Define the starting point
      x0 = np.array([-1.2, 1.0])
      
      # Set random seed (for reproducibility)
      np.random.seed(0)
      
      print("Demonstrate noise in function evaluation:")
      for i in range(5):
          print("objfun(x0) = %s" % str(rosenbrock_noisy(x0)))
      print("")
      
      # Call Py-BOBYQA
      soln = pybobyqa.solve(rosenbrock_noisy, x0)
      
      # Display output
      print(soln)
      
      # Compare with a derivative-based solver
      import scipy.optimize as opt
      soln = opt.minimize(rosenbrock_noisy, x0)
      
      print("")
      print("** SciPy results **")
      print("Solution xmin = %s" % str(soln.x))
      print("Objective value f(xmin) = %.10g" % (soln.fun))
      print("Needed %g objective evaluations" % soln.nfev)
      print("Exit flag = %g" % soln.status)
      print(soln.message)


The output of this is:

  .. code-block:: none
  
      Demonstrate noise in function evaluation:
      objfun(x0) = 24.6269006677
      objfun(x0) = 24.2968380444
      objfun(x0) = 24.4368545922
      objfun(x0) = 24.7422961542
      objfun(x0) = 24.6519490336
      
      ****** Py-BOBYQA Results ******
      Solution xmin = [-1.02866429  1.07341548]
      Objective value f(xmin) = 4.033118937
      Needed 36 objective evaluations (at 36 points)
      Approximate gradient = [-6921247.2999239  -3051622.27188687]
      Approximate Hessian = [[  1.98604897e+15   5.75929121e+14]
       [  5.75929121e+14   7.89533101e+14]]
      Exit flag = 0
      Success: rho has reached rhoend
      ******************************
      
      
      ** SciPy results **
      Solution xmin = [-1.2  1. ]
      Objective value f(xmin) = 23.80943672
      Needed 104 objective evaluations
      Exit flag = 2
      Desired error not necessarily achieved due to precision loss.

Although Py-BOBYQA does not find the true solution (and it cannot produce a good estimate of the objective gradient and Hessian), it still gives a reasonable decrease in the objective. However SciPy's derivative-based solver, which has no trouble solving the noise-free problem, is unable to make any progress.

As noted above, Py-BOBYQA has an input parameter :code:`objfun_has_noise` to indicate if :code:`objfun` has noise in it, which it does in this case. Therefore we can call Py-BOBYQA with

  .. code-block:: python
  
      soln = pybobyqa.solve(rosenbrock_noisy, x0, objfun_has_noise=True)

This time, we find the true solution, and better estimates of the gradient and Hessian:

  .. code-block:: none
  
      ****** Py-BOBYQA Results ******
      Solution xmin = [ 1.  1.]
      Objective value f(xmin) = 3.418770987e-18
      Needed 300 objective evaluations (at 300 points)
      Did a total of 4 runs
      Approximate gradient = [ -1.36175005e-08   2.12249758e-09]
      Approximate Hessian = [[ 805.93202374 -394.16671315]
       [-394.16671315  192.99451721]]
      Exit flag = 1
      Warning (max evals): Objective has been called MAXFUN times
      ******************************


References
----------

.. [CFMR2018]   
   C. Cartis, J. Fiala, B. Marteau and L. Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://arxiv.org/abs/1804.00154>`_, technical report, University of Oxford, (2018).

