contradictory-claims package README
===================================
A package for finding contradictory claims related to COVID-19 drug treatments in the CORD-19 literature

Installation
------------
To download this code and install in development mode, do the following:

.. code-block::
    
    $ fork a copy
    $ git clone https://github.com/CoronaWhy/drug-lit-contradictory-claims.git
    $ cd drug-lit-contradictory-claims
    $ pip install -e .
    
Dependencies
------------
.. code-block::

    $ pip install Torch:
       - Windows - pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
       - Mac - pip install torch torchvision 
            # MacOS Binaries dont support CUDA, install from source (https://pytorch.org/) if CUDA is needed
       - Linux - pip install torch torchvision


Testing |build| |coverage|
--------------------------
To test this code, please use ``tox``:

.. code-block::

    $ pip install tox
    $ tox

Note that ``tox`` is configured to automate running tests and checking test coverage, checking ``pyroma`` compliance,
checking ``flake8`` compliance, checking ``doc8`` compliance (for ``.rst`` files), enforcing README style guides, and
building ``sphinx`` documentation.

Documentation |documentation|
-----------------------------
Running ``tox`` above should automatically build the ``readTheDocs``-style ``sphinx`` documentation, however this can
also be accomplished by running the following:

.. code-block::

    $ cd docs
    $ make html
    $ open build/html/index.html

Usage
-----
This package is currently set up so that the training of the BERT model can be easily run as a package using a
command-line interface as follows:

.. code-block::

    $ # Make sure that installation was successful as described above
    $
    $ python -m contradictory_claims

.. |build| image:: https://travis-ci.com/CoronaWhy/drug-lit-contradictory-claims.svg?branch=master
    :target: https://travis-ci.com/CoronaWhy/drug-lit-contradictory-claims
    :alt: Build Status

.. |coverage| image:: https://codecov.io/gh/CoronaWhy/drug-lit-contradictory-claims/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/CoronaWhy/drug-lit-contradictory-claims

.. |documentation| image:: https://readthedocs.org/projects/drug-lit-contradictory-claims/badge/?version=latest
    :target: https://drug-lit-contradictory-claims.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
