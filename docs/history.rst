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

