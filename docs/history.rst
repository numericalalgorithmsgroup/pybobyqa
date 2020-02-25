Version History
===============
This section lists the different versions of Py-BOBYQA and the updates between them.

Version 1.0 (6 Feb 2018)
------------------------
* Initial release of Py-BOBYQA

Version 1.0.1 (20 Feb 2018)
---------------------------
* Minor bug fix to trust region subproblem solver (the output :code:`crvmin` is calculated correctly) - this has minimal impact on the performance of Py-BOBYQA.

Version 1.0.2 (20 Jun 2018)
---------------------------
* Extra optional input :code:`args` which passes through arguments for :code:`objfun` (pull request from `logangrado <https://github.com/logangrado>`_).
* Bug fixes: default parameters for reduced initialization cost regime, returning correct value from safety steps, retrieving dependencies during installation.

Version 1.1 (24 Dec 2018)
-------------------------
* Extra parameters to control the trust region radius over multiple restarts, designed for global optimization.
* New input flag :code:`seek_global_minimum` to set sensible default parameters for global optimization. New example script to demonstrate this functionality.
* Bug fix: default trust region radius when scaling variables within bounds.

Initially released as version 1.1a0 on 17 Jul 2018.

Version 1.1.1 (5 Apr 2019)
--------------------------
* Link code to Zenodo, to create DOI - no changes to the Py-BOBYQA algorithm.

Version 1.2 (25 Feb 2020)
-------------------------
* Use deterministic initialisation by default (so it is no longer necessary to set a random seed for reproducibility of Py-BOBYQA results).
* Full model Hessian stored rather than just upper triangular part - this improves the runtime of Hessian-based operations.
* Faster trust-region and geometry subproblem solutions in Fortran using the `trustregion <https://github.com/lindonroberts/trust-region>`_ package.
* Don’t adjust starting point if it is close to the bounds (as long as it is feasible).
* Option to stop default logging behavior and/or enable per-iteration printing.
* Bugfix: correctly handle 1-sided bounds as inputs, avoid divide-by-zero warnings when auto-detecting restarts.
