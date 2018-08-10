Overview
========

When to use Py-BOBYQA
---------------------
Py-BOBYQA is designed to solve the nonlinear least-squares minimization problem (with optional bound constraints)

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x)\\
   \text{s.t.} &\quad  a \leq x \leq b

We call :math:`f(x)` the objective function.

Py-BOBYQA is a *derivative-free* optimization algorithm, which means it does not require the user to provide the derivatives of :math:`f(x)`, nor does it attempt to estimate them internally (by using finite differencing, for instance). 

There are two main situations when using a derivative-free algorithm (such as Py-BOBYQA) is preferable to a derivative-based algorithm (which is the vast majority of least-squares solvers).

If **the residuals are noisy**, then calculating or even estimating their derivatives may be impossible (or at least very inaccurate). By noisy, we mean that if we evaluate :math:`f(x)` multiple times at the same value of :math:`x`, we get different results. This may happen when a Monte Carlo simulation is used, for instance, or :math:`f(x)` involves performing a physical experiment. 

If **the residuals are expensive to evaluate**, then estimating derivatives (which requires :math:`n` evaluations of :math:`f(x)` for every point of interest :math:`x`) may be prohibitively expensive. Derivative-free methods are designed to solve the problem with the fewest number of evaluations of the objective as possible.

**However, if you have provide (or a solver can estimate) derivatives** of :math:`f(x)`, then it is probably a good idea to use one of the many derivative-based solvers (such as `one from the SciPy library <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_).


Details of the Py-BOBYQA Algorithm
----------------------------------
Py-BOBYQA is a type of *trust-region* method, a common category of optimization algorithms for nonconvex problems. Given a current estimate of the solution :math:`x_k`, we compute a model which approximates the objective :math:`m_k(s)\approx f(x_k+s)` (for small steps :math:`s`), and maintain a value :math:`\Delta_k>0` (called the *trust region radius*) which measures the size of :math:`s` for which the approximation is good.

At each step, we compute a trial step :math:`s_k` designed to make our approximation :math:`m_k(s)` small (this task is called the *trust region subproblem*). We evaluate the objective at this new point, and if this provided a good decrease in the objective, we take the step (:math:`x_{k+1}=x_k+s_k`), otherwise we stay put (:math:`x_{k+1}=x_k`). Based on this information, we choose a new value :math:`\Delta_{k+1}`, and repeat the process.

In Py-BOBYQA, we construct our approximation :math:`m_k(s)` by interpolating a linear or quadratic approximation for :math:`f(x)` at several points close to :math:`x_k`. To make sure our interpolated model is accurate, we need to regularly check that the points are well-spaced, and move them if they aren't (i.e. improve the geometry of our interpolation points).

Py-BOBYQA is a Python implementation of the BOBYQA solver by Powell [Powell2009]_. More details about Py-BOBYQA algorithm are given in our paper [CFMR2018]_. 

References
----------

.. [CFMR2018]   
   C. Cartis, J. Fiala, B. Marteau and L. Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://arxiv.org/abs/1804.00154>`_, technical report, University of Oxford, (2018).

.. [Powell2009]   
   M. J. D. Powell, `The BOBYQA algorithm for bound constrained optimization without derivatives <http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf>`_, technical report DAMTP 2009/NA06, University of Cambridge, (2009).

