====================================================================
Py-BOBYQA: Derivative-Free Solver for Bound-Constrained Minimization
====================================================================

.. image::  https://github.com/numericalalgorithmsgroup/pybobyqa/actions/workflows/python_testing.yml/badge.svg
   :target: https://github.com/numericalalgorithmsgroup/pybobyqa/actions
   :alt: Build Status

.. image::  https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: GNU GPL v3 License

.. image:: https://img.shields.io/pypi/v/Py-BOBYQA.svg
   :target: https://pypi.python.org/pypi/Py-BOBYQA
   :alt: Latest PyPI version

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2630437.svg
   :target: https://doi.org/10.5281/zenodo.2630437
   :alt: DOI:10.5281/zenodo.2630437

.. image:: https://static.pepy.tech/personalized-badge/py-bobyqa?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads
 :target: https://pepy.tech/project/py-bobyqa
   :alt: Total downloads

Py-BOBYQA is a flexible package for solving bound-constrained general objective minimization, without requiring derivatives of the objective. At its core, it is a Python implementation of the BOBYQA algorithm by Powell, but Py-BOBYQA has extra features improving its performance on some problems (see the papers below for details). Py-BOBYQA is particularly useful when evaluations of the objective function are expensive and/or noisy.

More details about Py-BOBYQA and its enhancements over BOBYQA can be found in our papers:

1. Coralia Cartis, Jan Fiala, Benjamin Marteau and Lindon Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41 [`arXiv preprint: 1804.00154 <https://arxiv.org/abs/1804.00154>`_] 
2. Coralia Cartis, Lindon Roberts and Oliver Sheridan-Methven, `Escaping local minima with derivative-free methods: a numerical investigation <https://doi.org/10.1080/02331934.2021.1883015>`_, *Optimization*, 71:8 (2022), pp. 2343-2373. [`arXiv preprint: 1812.11343 <https://arxiv.org/abs/1812.11343>`_] 
3. Lindon Roberts, `Model Construction for Convex-Constrained Derivative-Free Optimization <https://arxiv.org/abs/2403.14960>`_, *arXiv preprint arXiv:2403.14960* (2024).

Please cite [1] when using Py-BOBYQA for local optimization, [1,2] when using Py-BOBYQA's global optimization heuristic functionality, and [1,3] if using the general convex constraints functionality.

The original paper by Powell is: M. J. D. Powell, The BOBYQA algorithm for bound constrained optimization without derivatives, technical report DAMTP 2009/NA06, University of Cambridge (2009), and the original Fortran implementation is available `here <http://mat.uc.pt/~zhang/software.html>`_.

If you are interested in solving least-squares minimization problems, you may wish to try `DFO-LS <https://github.com/numericalalgorithmsgroup/dfols>`_, which has the same features as Py-BOBYQA (plus some more), and exploits the least-squares problem structure, so performs better on such problems.

Documentation
-------------
See manual.pdf or the `online manual <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_.

Citation
--------
Full details of the Py-BOBYQA algorithm are given in our papers: 

1. Coralia Cartis, Jan Fiala, Benjamin Marteau and Lindon Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41 [`preprint <https://arxiv.org/abs/1804.00154>`_] 
2. Coralia Cartis, Lindon Roberts and Oliver Sheridan-Methven, `Escaping local minima with derivative-free methods: a numerical investigation <https://doi.org/10.1080/02331934.2021.1883015>`_, *Optimization*, 71:8 (2022), pp. 2343-2373. [`arXiv preprint: 1812.11343 <https://arxiv.org/abs/1812.11343>`_]
3. Lindon Roberts, `Model Construction for Convex-Constrained Derivative-Free Optimization <https://arxiv.org/abs/2403.14960>`_, *arXiv preprint arXiv:2403.14960* (2024).

Please cite [1] when using Py-BOBYQA for local optimization, [1,2] when using Py-BOBYQA's global optimization heuristic functionality, and [1,3] if using the general convex constraints functionality.

Requirements
------------
Py-BOBYQA requires the following software to be installed:

* Python 3.8 or higher (http://www.python.org/)

Additionally, the following python packages should be installed (these will be installed automatically if using *pip*, see `Installation using pip`_):

* NumPy (http://www.numpy.org/)
* SciPy (http://www.scipy.org/)
* Pandas (http://pandas.pydata.org/)

**Optional package:** Py-BOBYQA versions 1.2 and higher also support the `trustregion <https://github.com/lindonroberts/trust-region>`_ package for fast trust-region subproblem solutions. To install this, make sure you have a Fortran compiler (e.g. `gfortran <https://gcc.gnu.org/wiki/GFortran>`_) and NumPy installed, then run :code:`pip install trustregion`. You do not have to have trustregion installed for Py-BOBYQA to work, and it is not installed by default.

Installation using pip
----------------------
For easy installation, use `pip <http://www.pip-installer.org/>`_:

 .. code-block:: bash

    $ pip install Py-BOBYQA

Note that if an older install of Py-BOBYQA is present on your system you can use:

 .. code-block:: bash

    $ pip install --upgrade Py-BOBYQA

to upgrade Py-BOBYQA to the latest version.

Manual installation
-------------------
Alternatively, you can download the source code from `Github <https://github.com/numericalalgorithmsgroup/pybobyqa>`_ and unpack as follows:

 .. code-block:: bash

    $ git clone https://github.com/numericalalgorithmsgroup/pybobyqa
    $ cd pybobyqa

Py-BOBYQA is written in pure Python and requires no compilation. It can be installed using:

 .. code-block:: bash

    $ pip install .

instead.

To upgrade Py-BOBYQA to the latest version, navigate to the top-level directory (i.e. the one containing :code:`setup.py`) and rerun the installation using :code:`pip`, as above:

 .. code-block:: bash

    $ git pull
    $ pip install .

Testing
-------
If you installed Py-BOBYQA manually, you can test your installation using the pytest package:

 .. code-block:: bash

    $ pip install pytest
    $ python -m pytest --pyargs pybobyqa

Alternatively, the HTML documentation provides some simple examples of how to run Py-BOBYQA.

Examples
--------
Examples of how to run Py-BOBYQA are given in the `online documentation <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_, and the `examples directory <https://github.com/numericalalgorithmsgroup/pybobyqa/tree/master/examples>`_ in Github.

Uninstallation
--------------
If Py-BOBYQA was installed using *pip* you can uninstall as follows:

 .. code-block:: bash

    $ pip uninstall Py-BOBYQA

If Py-BOBYQA was installed manually you have to remove the installed files by hand (located in your python site-packages directory).

Bugs
----
Please report any bugs using GitHub's issue tracker.

License
-------
This algorithm is released under the GNU GPL license. Please `contact NAG <http://www.nag.com/content/worldwide-contact-information>`_ for alternative licensing.
