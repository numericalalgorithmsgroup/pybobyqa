Installing Py-BOBYQA
====================

Requirements
------------
Py-BOBYQA requires the following software to be installed:

* Python 3.8 or higher (http://www.python.org/)

Additionally, the following python packages should be installed (these will be installed automatically if using `pip <http://www.pip-installer.org/>`_, see `Installation using pip`_):

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
The source code for Py-BOBYQA is `available on Github <https://github.com/numericalalgorithmsgroup/pybobyqa>`_:

 .. code-block:: bash
 
    $ git clone https://github.com/numericalalgorithmsgroup/pybobyqa
    $ cd pybobyqa

Py-BOBYQA is written in pure Python and requires no compilation. It can be installed using:

 .. code-block:: bash

    $ pip install .

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

Uninstallation
--------------
If Py-BOBYQA was installed using `pip <http://www.pip-installer.org/>`_ you can uninstall as follows:

 .. code-block:: bash

    $ pip uninstall Py-BOBYQA

If Py-BOBYQA was installed manually you have to remove the installed files by hand (located in your python site-packages directory).


