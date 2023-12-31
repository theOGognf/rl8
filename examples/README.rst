Setup
=====

Install the full version of MLflow to make the most of **rl8**'s
integration with its experiment tracking.

.. code:: console

    pip install mlflow

Running Examples
================

Example main scripts (``__main__.py``) are intended to be ran as modules
from the **rl8** project root directory as follows (assuming ``$EXAMPLE``
is the example directory you want to run):

.. code:: console

    python -m examples.$EXAMPLE

If this is your first time using **rl8**, MLflow will create an ``mlruns``
directory within **rl8**'s project root directory when you run an example.
The ``mlruns`` directory will contain an experiment directory (each example
is ran as an MLflow experiment), under which will be an example's run directories.
You can track experiment progress using the MLflow UI:

.. code:: console

    mlflow ui
