AlgoTrading
===========

An example containing a custom environment with three custom models that
showcase the speed and utility of **rlstack** when it comes to learning
complex behaviors based on historical data.

The environment simulates an asset's price according to the equation
``y[k + 1] = (1 + km) * (1 + kc * sin(f * t)) * y[k]`` where
``km``, ``kc``, ``f``, and ``y[0]`` are all randomly sampled
from their own independent uniform distributions, some of which
are defined by values in the environment's config.

A policy must learn to hold, buy, or sell the asset based on the
asset's change in price with respect to the previous day and with
respect to the price at which the policy had previously bought the
asset. The parameterization of the asset's price makes learning to
trade the asset a bit difficult if historical data isn't used as inputs
to the policy's model.

All custom models provided also use action masking to ignore impossible actions
according to the environment's observation (i.e., disallowing buying
an asset when the asset is already owned).

Organization
------------

It's recommended that you browse the example's files to gain an understanding
of how custom environments and models should be defined to get the most
benefit out of **rlstack**'s design. The example is organized as follows:

* ``./__main__.py`` is the main script.
* ``./env.py`` contains the environment's definition.
* ``./models`` contains model definitions.

Running the Example
-------------------

The example's main script is intended to be ran as a module and CLI from the
**rlstack** project root directory as follows:

.. code:: console

    python -m examples.algotrading

If this is your first time using **rlstack**, MLFlow will create
an ``../mlruns`` directory when you run the example. The ``../mlruns``
directory will contain an experiment directory (where multiple example
runs can reside), under which will be the example's run directory. You can
track experiment progress by reloading files within the run directory, or by
using the MLFlow UI by running ``mlflow ui`` while in ``../mlruns``'s parent
directory.
