Using Py-BOBYQA
===============
This section describes the main interface to Py-BOBYQA and how to use it.

Nonlinear Minimization
----------------------
Py-BOBYQA is designed to solve the local optimization problem

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x) \\
   \text{s.t.} &\quad  a \leq x \leq b

where the bound constraints :math:`a \leq x \leq b` are optional. The upper and lower bounds on the variables are non-relaxable (i.e. Py-BOBYQA will never ask to evaluate a point outside the bounds). The objective function :math:`f(x)` is usually nonlinear and nonquadratic. If you know your objective is linear or quadratic, you should consider a solver designed for such functions (see `here <https://neos-guide.org/Optimization-Guide>`_ for details).

Py-BOBYQA iteratively constructs an interpolation-based model for the objective, and determines a step using a trust-region framework.
For an in-depth technical description of the algorithm see the paper [CFMR2018]_, and for the global optimization heuristic, see [CRO2018]_.

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
                  seek_global_minimum=False, 
                  scaling_within_bounds=False,
                  do_logging=True, print_progress=False)

These arguments are:

* :code:`args` - a tuple of extra arguments passed to the objective function.
* :code:`bounds` - a tuple :code:`(lower, upper)` with the vectors :math:`a` and :math:`b` of lower and upper bounds on :math:`x` (default is :math:`a_i=-10^{20}` and :math:`b_i=10^{20}`). To set bounds for either :code:`lower` or :code:`upper`, but not both, pass a tuple :code:`(lower, None)` or :code:`(None, upper)`.
* :code:`npt` - the number of interpolation points to use (default is :math:`2n+1` for a problem with :code:`len(x0)=n` if :code:`objfun_has_noise=False`, otherwise it is set to :math:`(n+1)(n+2)/2`). Py-BOBYQA requires :math:`n+1 \leq npt \leq (n+1)(n+2)/2`. Larger values are particularly useful for noisy problems.
* :code:`rhobeg` - the initial value of the trust region radius (default is 0.1 if :code:`scaling_within_bounds=True`, otherwise :math:`0.1\max(\|x_0\|_{\infty}, 1)`).
* :code:`rhoend` - minimum allowed value of trust region radius, which determines when a successful termination occurs (default is :math:`10^{-8}`).
* :code:`maxfun` - the maximum number of objective evaluations the algorithm may request (default is :math:`\min(100(n+1),1000)`).
* :code:`nsamples` - a Python function :code:`nsamples(delta, rho, iter, nrestarts)` which returns the number of times to evaluate :code:`objfun` at a given point. This is only applicable for objectives with stochastic noise, when averaging multiple evaluations at the same point produces a more accurate value. The input parameters are the trust region radius (:code:`delta`), the lower bound on the trust region radius (:code:`rho`), how many iterations the algorithm has been running for (:code:`iter`), and how many restarts have been performed (:code:`nrestarts`). Default is no averaging (i.e. :code:`nsamples(delta, rho, iter, nrestarts)=1`).
* :code:`user_params` - a Python dictionary :code:`{'param1': val1, 'param2':val2, ...}` of optional parameters. A full list of available options is given in the next section :doc:`advanced`.
* :code:`objfun_has_noise` - a flag to indicate whether or not :code:`objfun` has stochastic noise; i.e. will calling :code:`objfun(x)` multiple times at the same value of :code:`x` give different results? This is used to set some sensible default parameters (including using multiple restarts), all of which can be overridden by the values provided in :code:`user_params`.
* :code:`seek_global_minimum` - a flag to indicate whether to search for a global minimum, rather than a local minimum. This is used to set some sensible default parameters, all of which can be overridden by the values provided in :code:`user_params`. If :code:`True`, both upper and lower bounds must be set. Note that Py-BOBYQA only implements a heuristic method, so there are no guarantees it will find a global minimum. However, by using this flag, it is more likely to escape local minima if there are better values nearby. The method used is a multiple restart mechanism, where we repeatedly re-initialize Py-BOBYQA from the best point found so far, but where we use a larger trust reigon radius each time (note: this is different to more common multi-start approach to global optimization).
* :code:`scaling_within_bounds` - a flag to indicate whether the algorithm should internally shift and scale the entries of :code:`x` so that the bounds become :math:`0 \leq x \leq 1`. This is useful is you are setting :code:`bounds` and the bounds have different orders of magnitude. If :code:`scaling_within_bounds=True`, the values of :code:`rhobeg` and :code:`rhoend` apply to the *shifted* variables.
* :code:`do_logging` - a flag to indicate whether logging output should be produced. This is not automatically visible unless you use the Python `logging <https://docs.python.org/3/library/logging.html>`_ module (see below for simple usage).
* :code:`print_progress` - a flag to indicate whether to print a per-iteration progress log to terminal.

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
      
      # Call Py-BOBYQA
      soln = pybobyqa.solve(rosenbrock, x0)
      
      # Display output
      print(soln)
      
Note that Py-BOBYQA is a randomized algorithm: in its first phase, it builds an internal approximation to the objective function by sampling it along random directions. In the code above, we set NumPy's random seed for reproducibility over multiple runs, but this is not required. The output of this script, showing that Py-BOBYQA finds the correct solution, is

  .. code-block:: none
  
      ****** Py-BOBYQA Results ******
      Solution xmin = [1. 1.]
      Objective value f(xmin) = 1.013856052e-20
      Needed 151 objective evaluations (at 151 points)
      Approximate gradient = [ 2.35772499e-08 -1.07598803e-08]
      Approximate Hessian = [[ 802.00799968 -400.04089119]
       [-400.04089119  199.99228723]]
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
      Solution xmin = [0.9  0.81]
      Objective value f(xmin) = 0.01
      Needed 146 objective evaluations (at 146 points)
      Approximate gradient = [-2.00000006e-01 -4.74578563e-09]
      Approximate Hessian = [[ 649.66398033 -361.03094781]
       [-361.03094781  199.94223213]]
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
      Initialising (coordinate directions)
      Function eval 2 at point 2 has f = 14.337296 at x = [-1.08  0.85]
      Function eval 3 at point 3 has f = 55.25 at x = [-1.2   0.73]
      ...
      Function eval 145 at point 145 has f = 0.0100000013172792 at x = [0.89999999 0.80999999]
      Function eval 146 at point 146 has f = 0.00999999999999993 at x = [0.9  0.81]
      Did a total of 1 run(s)

If we wanted to save this output to a file, we could replace the above call to :code:`logging.basicConfig()` with

  .. code-block:: python
  
      logging.basicConfig(filename="myfile.log", level=logging.INFO, 
                          format='%(message)s', filemode='w')

If you have logging for some parts of your code and you want to deactivate all Py-BOBYQA logging, you can use the optional argument :code:`do_logging=False` in :code:`pybobyqa.solve()`.

An alternative option available is to get Py-BOBYQA to print to terminal progress information every iteration, by setting the optional argument :code:`print_progress=True` in :code:`pybobyqa.solve()`. If we do this for the above example, we get

  .. code-block:: none
  
       Run  Iter     Obj       Grad     Delta      rho     Evals 
        1     1    1.43e+01  1.74e+02  1.20e-01  1.20e-01    5   
        1     2    5.57e+00  1.20e+02  3.66e-01  1.20e-01    6   
        1     3    5.57e+00  1.20e+02  6.00e-02  1.20e-02    6
      ...
        1    132   1.00e-02  2.00e-01  1.50e-08  1.00e-08   144  
        1    133   1.00e-02  2.00e-01  1.50e-08  1.00e-08   145  

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
          print("objfun(x0) = %g" % rosenbrock_noisy(x0))
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
      objfun(x0) = 24.6269
      objfun(x0) = 24.2968
      objfun(x0) = 24.4369
      objfun(x0) = 24.7423
      objfun(x0) = 24.6519
      
      ****** Py-BOBYQA Results ******
      Solution xmin = [-1.04327395  1.09935385]
      Objective value f(xmin) = 4.080030471
      Needed 42 objective evaluations (at 42 points)
      Approximate gradient = [-3786376.5065785   5876675.51763198]
      Approximate Hessian = [[ 1.32881117e+14 -2.68241358e+14]
       [-2.68241358e+14  6.09785319e+14]]
      Exit flag = 0
      Success: rho has reached rhoend
      ******************************
      
      
      ** SciPy results **
      Solution xmin = [-1.20013817  0.99992915]
      Objective value f(xmin) = 23.86371763
      Needed 80 objective evaluations
      Exit flag = 2
      Desired error not necessarily achieved due to precision loss.

Although Py-BOBYQA does not find the true solution (and it cannot produce a good estimate of the objective gradient and Hessian), it still gives a reasonable decrease in the objective. However SciPy's derivative-based solver, which has no trouble solving the noise-free problem, is unable to make any progress.

As noted above, Py-BOBYQA has an input parameter :code:`objfun_has_noise` to indicate if :code:`objfun` has noise in it, which it does in this case. Therefore we can call Py-BOBYQA with

  .. code-block:: python
  
      soln = pybobyqa.solve(rosenbrock_noisy, x0, objfun_has_noise=True)

This time, we find the true solution, and better estimates of the gradient and Hessian:

  .. code-block:: none
  
      ****** Py-BOBYQA Results ******
      Solution xmin = [1. 1.]
      Objective value f(xmin) = 1.237351799e-19
      Needed 300 objective evaluations (at 300 points)
      Did a total of 5 runs
      Approximate gradient = [-2.17072738e-07  9.62304351e-08]
      Approximate Hessian = [[ 809.56521044 -400.33737779]
       [-400.33737779  198.36487985]]
      Exit flag = 1
      Warning (max evals): Objective has been called MAXFUN times
      ******************************


Example: Global Optimization
----------------------------
The following example shows how to use the global optimization features of Py-BOBYQA. Here, we try to minimize the Freudenstein and Roth function (problem 2 in J.J. Moré, B.S. Garbow, B.S. and K.E. Hillstrom, Testing Unconstrained Optimization Software, *ACM Trans. Math. Software* 7:1 (1981), 17-41). This function has two local minima, one of which is global.

Note that Py-BOBYQA only implements a heuristic method, so there are no guarantees it will find a global minimum. However, by using the :code:`seek_global_minimum` flag, it is more likely to escape local minima if there are better values nearby.

  .. code-block:: python
  
      # Py-BOBYQA example: globally minimize the Freudenstein and Roth function
      from __future__ import print_function
      import numpy as np
      import pybobyqa
      
      # Define the objective function
      # This function has a local minimum f = 48.98 
      # at x = np.array([11.41, -0.8968])
      # and a global minimum f = 0 at x = np.array([5.0, 4.0])
      def freudenstein_roth(x):
          r1 = -13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1]
          r2 = -29.0 + x[0] + ((1.0 + x[1]) * x[1] - 14.0) * x[1]
          return r1 ** 2 + r2 ** 2
      
      # Define the starting point
      x0 = np.array([5.0, -20.0])
      
      # Define bounds (required for global optimization)
      lower = np.array([-30.0, -30.0])
      upper = np.array([30.0, 30.0])
      
      # Set random seed (for reproducibility)
      np.random.seed(0)
      
      print("First run - search for local minimum only")
      print("")
      soln = pybobyqa.solve(freudenstein_roth, x0, maxfun=500, 
                            bounds=(lower, upper))
      print(soln)
      
      print("")
      print("")
      
      print("Second run - search for global minimum")
      print("")
      soln = pybobyqa.solve(freudenstein_roth, x0, maxfun=500, 
                            bounds=(lower, upper), 
                            seek_global_minimum=True)
      print(soln)

The output of this is:

  .. code-block:: none
  
      First run - search for local minimum only
      
      ****** Py-BOBYQA Results ******
      Solution xmin = [11.41277902 -0.89680525]
      Objective value f(xmin) = 48.98425368
      Needed 143 objective evaluations (at 143 points)
      Approximate gradient = [-1.64941396e-07  9.69795781e-07]
      Approximate Hessian = [[   7.74717421 -104.51102613]
       [-104.51102613 1135.76500421]]
      Exit flag = 0
      Success: rho has reached rhoend
      ******************************
      
      
      
      Second run - search for global minimum
      
      ****** Py-BOBYQA Results ******
      Solution xmin = [5. 4.]
      Objective value f(xmin) = 3.659891409e-17
      Needed 500 objective evaluations (at 500 points)
      Did a total of 5 runs
      Approximate gradient = [ 8.70038835e-10 -4.64918043e-07]
      Approximate Hessian = [[   4.28883646   64.16836253]
       [  64.16836253 3722.93109385]]
      Exit flag = 1
      Warning (max evals): Objective has been called MAXFUN times
      ******************************

As we can see, the :code:`seek_global_minimum` flag helped Py-BOBYQA escape the local minimum from the first run, and find the global minimum. More details are given in [CRO2018]_.

References
----------

.. [CFMR2018]   
   Coralia Cartis, Jan Fiala, Benjamina Marteau and Lindon Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41 [`preprint <https://arxiv.org/abs/1804.00154>`_] 
.. [CRO2018]   
   Coralia Cartis, Lindon Roberts and Oliver Sheridan-Methven, `Escaping local minima with derivative-free methods: a numerical investigation <https://doi.org/10.1080/02331934.2021.1883015>`_, *Optimization* (2021). [`preprint <https://arxiv.org/abs/1812.11343>`_] 
