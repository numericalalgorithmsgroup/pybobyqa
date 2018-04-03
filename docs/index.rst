.. Py-BOBYQA documentation master file, created by
   sphinx-quickstart on Wed Nov  8 10:59:20 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Py-BOBYQA: Derivative-Free Optimizer for Bound-Constrained Minimization
=======================================================================

**Release:** |version|

**Date:** |today|

**Author:** `Lindon Roberts <lindon.roberts@maths.ox.ac.uk>`_

Py-BOBYQA is a flexible package for finding local solutions to nonlinear, nonconvex minimization problems (with optional bound constraints), without requiring any derivatives of the objective. Py-BOBYQA is a Python implementation of the `BOBYQA <http://mat.uc.pt/~zhang/software.html#powell_software>`_ solver by Powell (documentation `here <http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf>`_). It is particularly useful when evaluations of the objective function are expensive and/or noisy.

That is, Py-BOBYQA solves

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x)\\
   \text{s.t.} &\quad  a \leq x \leq b

Full details of the Py-BOBYQA algorithm are given in our paper: C. Cartis, J. Fiala, B. Marteau and L. Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://arxiv.org/abs/1804.00154>`_, technical report, University of Oxford, (2018).

If you are interested in solving least-squares minimization problems, you may wish to try `DFO-LS <https://github.com/numericalalgorithmsgroup/dfols>`_, which has the same features as Py-BOBYQA (plus some more), and exploits the least-squares problem structure, so performs better on such problems.

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
This software was developed under the supervision of `Coralia Cartis <https://www.maths.ox.ac.uk/people/coralia.cartis>`_, and was supported by the EPSRC Centre For Doctoral Training in `Industrially Focused Mathematical Modelling <https://www.maths.ox.ac.uk/study-here/postgraduate-study/industrially-focused-mathematical-modelling-epsrc-cdt>`_ (EP/L015803/1) in collaboration with the `Numerical Algorithms Group <http://www.nag.com/>`_.

