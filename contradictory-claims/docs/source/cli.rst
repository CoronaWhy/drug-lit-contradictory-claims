Command Line Interface
======================
.. note:: The command line wrapper might not work on Windows. Use :code:`python3 -m contradictory_claims`
    if it has issues.

``contradictory-claims`` automatically installs the command :code:`contradictory-claims`. This command can be used
to use the :func:`contradictory_claims.models.train_model.train_model` function via the command line for
default training settings.

.. click:: contradictory_claims.cli:main
   :prog: contradictory-claims
   :show-nested:
