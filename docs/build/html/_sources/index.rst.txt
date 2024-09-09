.. Py-BOBYQA documentation master file, created by
   sphinx-quickstart on Wed Nov  8 10:59:20 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Py-BOBYQA: Derivative-Free Optimizer for Bound-Constrained Minimization
=======================================================================

**Release:** |version|

**Date:** |today|

**Author:** `Lindon Roberts <lindon.roberts@sydney.edu.au>`_

Py-BOBYQA is a flexible package for finding local solutions to nonlinear, nonconvex minimization problems (with optional bound and other convex constraints), without requiring any derivatives of the objective. Py-BOBYQA is a Python implementation of the `BOBYQA <http://mat.uc.pt/~zhang/software.html#powell_software>`_ solver by Powell (documentation `here <http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf>`_). It is particularly useful when evaluations of the objective function are expensive and/or noisy.

That is, Py-BOBYQA solves

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x)\\
   \text{s.t.} &\quad  a \leq x \leq b \\
   &\quad x \in C := C_1 \cap \cdots \cap C_n, \quad \text{all $C_i$ convex}

If provided, the constraints the variables are non-relaxable (i.e. Py-BOBYQA will never ask to evaluate a point outside the bounds),
although the general :math:`x \in C` constraint may be slightly violated from rounding errors.

Full details of the Py-BOBYQA algorithm are given in our papers: 

1. Coralia Cartis, Jan Fiala, Benjamin Marteau and Lindon Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41 [`preprint <https://arxiv.org/abs/1804.00154>`_] 
2. Coralia Cartis, Lindon Roberts and Oliver Sheridan-Methven, `Escaping local minima with derivative-free methods: a numerical investigation <https://doi.org/10.1080/02331934.2021.1883015>`_, *Optimization*, 71:8 (2022), pp. 2343-2373. [`arXiv preprint: 1812.11343 <https://arxiv.org/abs/1812.11343>`_]
3. Lindon Roberts, `Model Construction for Convex-Constrained Derivative-Free Optimization <https://arxiv.org/abs/2403.14960>`_, *arXiv preprint arXiv:2403.14960* (2024).

Please cite [1] when using Py-BOBYQA for local optimization, [1,2] when using Py-BOBYQA's global optimization heuristic functionality, and [1,3] if using the general convex constraints :math:`x \in C` functionality.

If you are interested in solving least-squares minimization problems, you may wish to try `DFO-LS <https://github.com/numericalalgorithmsgroup/dfols>`_, which has the same features as Py-BOBYQA (plus some more), and exploits the least-squares problem structure, so performs better on such problems.

Since v1.1, Py-BOBYQA has a heuristic for global optimization (see :doc:`userguide` for details). As this is a heuristic, there are no guarantees it will find a global minimum, but it is more likely to escape local minima if there are better values nearby.

Py-BOBYQA is released under the GNU General Public License. Please `contact NAG <http://www.nag.com/content/worldwide-contact-information>`_ for alternative licensing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   info
   userguide
   advanced
   diagnostic
   history

Acknowledgements
----------------
This software was initially developed under the supervision of `Coralia Cartis <https://www.maths.ox.ac.uk/people/coralia.cartis>`_, and was supported by the EPSRC Centre For Doctoral Training in `Industrially Focused Mathematical Modelling <https://www.maths.ox.ac.uk/study-here/postgraduate-study/industrially-focused-mathematical-modelling-epsrc-cdt>`_ (EP/L015803/1) in collaboration with the `Numerical Algorithms Group <http://www.nag.com/>`_.
Development of Py-BOBYQA has also been supported by the Australian Research Council (DE240100006).

