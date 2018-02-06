Installing Py-BOBYQA
====================

Requirements
------------
Py-BOBYQA requires the following software to be installed:

* `Python 2.7 or Python 3 <http://www.python.org/>`_

Additionally, the following python packages should be installed (these will be installed automatically if using `pip <http://www.pip-installer.org/>`_, see `Installation using pip`_):

* `NumPy 1.11 or higher <http://www.numpy.org/>`_ 
* `SciPy 0.18 or higher <http://www.scipy.org/>`_
* `Pandas 0.17 or higher <https://pandas.pydata.org/>`_


Installation using pip
----------------------
For easy installation, use `pip <http://www.pip-installer.org/>`_ as root:

 .. code-block:: bash

    $ [sudo] pip install Py-BOBYQA

If you do not have root privileges or you want to install Py-BOBYQA for your private use, you can use:

 .. code-block:: bash

    $ pip install --user Py-BOBYQA
      
which will install Py-BOBYQA in your home directory.

Note that if an older install of Py-BOBYQA is present on your system you can use:

 .. code-block:: bash

    $ [sudo] pip install --upgrade Py-BOBYQA
      
to upgrade Py-BOBYQA to the latest version.

Manual installation
-------------------
The source code for Py-BOBYQA is `available on Github <https://https://github.com/numericalalgorithmsgroup/pybobyqa>`_:

 .. code-block:: bash
 
    $ git clone https://github.com/numericalalgorithmsgroup/pybobyqa
    $ cd pybobyqa

Py-BOBYQA is written in pure Python and requires no compilation. It can be installed using:

 .. code-block:: bash

    $ [sudo] pip install .

If you do not have root privileges or you want to install Py-BOBYQA for your private use, you can use:

 .. code-block:: bash

    $ pip install --user .
    
instead.    

Testing
-------
If you installed Py-BOBYQA manually, you can test your installation by running:

 .. code-block:: bash

    $ python setup.py test

Uninstallation
--------------
If Py-BOBYQA was installed using `pip <http://www.pip-installer.org/>`_ you can uninstall as follows:

 .. code-block:: bash

    $ [sudo] pip uninstall Py-BOBYQA

If Py-BOBYQA was installed manually you have to remove the installed files by hand (located in your python site-packages directory).


