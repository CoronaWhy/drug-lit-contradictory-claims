contradictory-claims package README
===================================
A package for finding contradictory claims related to COVID-19 drug treatments in the CORD-19 literature

Installation
------------
To download this code and install in development mode, do the following:

.. code-block::

    $ git clone https://github.com/CoronaWhy/drug-lit-contradictory-claims.git
    $ cd drug-lit-contradictory-claims
    $ pip install -e .

Testing
-------
To test this code, please use ``tox``:

.. code-block::

    $ pip install tox
    $ tox

Note that ``tox`` is configured to automate running tests and checking test coverage, checking ``pyroma`` compliance,
checking ``flake8`` compliance, checking ``doc8`` compliance (for ``.rst`` files), enforcing README style guides, and
building ``sphinx`` documentation.

Documentation
-------------
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
