Running Examples
================

Example main scripts (``__main__.py``) are intended to be ran as modules and
CLIs from the **rl8** project root directory as follows (assuming
``$EXAMPLE`` is the example directory you want to run):

.. code:: console

    python -m examples.$EXAMPLE

If this is your first time using **rl8**, MLflow will create an ``mlruns``
directory within **rl8**'s project root directory when you run an example.
The ``mlruns`` directory will contain an experiment directory (each example
is ran as an MLflow experiment), under which will be an example's run directories.
You can track experiment progress by reloading files within a run directory, or
by using the MLflow UI by running ``mlflow ui`` while in ``mlruns``'s parent
directory (i.e., **rl8**'s project root directory).
