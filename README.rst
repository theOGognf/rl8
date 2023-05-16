rlstack: A Minimal RL Toolkit
=============================

**rlstack** is a high throughput, infinite horizon RL library that can
simulate highly parallelized environments, and can train a PPO policy using
those highly parallelized environments, achieving around 500k environment
transitions (and one policy update) per second using a single, off-the-shelf
computing device.

* **Documentation**: https://theogognf.github.io/rlstack/
* **PyPI**: https://pypi.org/project/rlstack/
* **Repository**: https://github.com/theOGognf/rlstack

Quick Start
===========

Installation
------------

Install with pip for the latest (stable) version.

.. code:: console

    pip install rlstack

Install from GitHub for the latest (unstable) version.

.. code:: console

    git clone https://github.com/theOGognf/rlstack.git
    pip install ./rlstack/

Basic Usage
-----------

Collect environment transitions and update a policy directly using the
low-level algorithm interface (this updates the policy once).

.. code:: python

    from rlstack import Algorithm
    from rlstack.env import DiscreteDummyEnv

    algo = Algorithm(DiscreteDummyEnv)
    algo.collect()
    algo.step()

Train a policy with PPO and log training progress with MLFlow using the
high-level trainer interface (this updates the policy infinitely).

.. code:: python

    from rlstack import Trainer
    from rlstack.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv)
    trainer.run()

Quick Examples
==============

Customizing Training Runs
-------------------------

Use a custom distribution and custom hyperparameter values with the low-level
algorithm interface. The feedforward algorithm uses default feedforward models
for different environment action specs.

.. code:: python

    from rlstack import Algorithm, SquashedNormal
    from rlstack.env import ContinuousDummyEnv

    algo = Algorithm(
        ContinuousDummyEnv,
        distribution_cls=SquashedNormal,
        gae_lambda=0.99,
        gamma=0.99,
    )
    algo.collect()
    algo.step()

Use the same custom distribution and custom hyperparameter values with the
high-level trainer interface.

.. code:: python

    from rlstack import SquashedNormal, Trainer
    from rlstack.env import ContinuousDummyEnv

    trainer = Trainer(
        ContinuousDummyEnv,
        algorithm_config={
            "distribution_cls": SquashedNormal,
            "gae_lambda": 0.99,
            "gamma": 0.99,
        }
    )
    trainer.run()

Training a Recurrent Policy
---------------------------

Use the low-level algorithm interface to seamlessly switch between feedforward
and recurrent algorithms. The recurrent algorithm uses default recurrent models
for different environment action specs.

.. code:: python

    from rlstack import RecurrentAlgorithm
    from rlstack.env import DiscreteDummyEnv

    algo = RecurrentAlgorithm(DiscreteDummyEnv)
    algo.collect()
    algo.step()

Specify the algorithm type using the high-level trainer interface (which
defaults to a feedforward algorithm).

.. code:: python

    from rlstack import RecurrentAlgorithm, Trainer
    from rlstack.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv, algorithm_cls=RecurrentAlgorithm)
    trainer.run()

Training on a GPU
-----------------

Use the low-level algorithm interface to specify training on a GPU.

.. code:: python

    from rlstack import Algorithm
    from rlstack.env import DiscreteDummyEnv

    algo = Algorithm(DiscreteDummyEnv, device="cuda")
    algo.collect()
    algo.step()

Use the high-level trainer interface to specify training on a GPU.

.. code:: python

    from rlstack import Trainer
    from rlstack.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv, algorithm_config={"device": "cuda"})
    trainer.run()

Specifying Training Stop Conditions
-----------------------------------

Use the high-level trainer interface to specify training stop conditions based
on training statistics.

.. code:: python

    from rlstack import Trainer
    from rlstack.conditions import Plateaus
    from rlstack.env import DiscreteDummyEnv

    trainer = Trainer(
        DiscreteDummyEnv,
        stop_conditions=[Plateaus("returns/mean", rtol=0.05)],
    )
    trainer.run()

Why rlstack?
============

There are many high quality, open-sourced RL libraries. Most of them take on the
daunting task of being a monolithic, one-stop-shop for everything RL, attempting to
support as many algorithms, environments, models, and compute capabilities as possible.
Naturely, this monolothic goal has some drawbacks:

* The software becomes more dense with each supported feature, making the library
  all-the-more difficult to customize for a specific use case.
* The software becomes less performant for a specific use case. RL practitioners
  typically end up accepting the cost of transitioning to expensive and
  difficult-to-manage compute clusters to get results faster.

There's a handful of high quality, open-sourced RL libraries that tradeoff feature
richness to reduce these drawbacks. However, each library still doesn't provide
enough speed benefit to warrant the switch from a monolithic repo, or is still
too complex to adapt to a specific use case.

**rlstack** is a niche RL library that finds a goldilocks zone between the
feature support and speed/complexity tradeoff by making some key assumptions:

* Environments are highly parallelized and their parallelization is entirely
  managed within the environment. This allows **rlstack** to ignore distributed
  computing design considerations.
* Environments are infinite horizon (i.e., they have no terminal conditions).
  This allows **rlstack** to reset environments at the same, fixed horizon
  intervals, greatly simplifying environment and algorithm implementations.
* The only supported ML framework is PyTorch and the only supported algorithm
  is PPO. This allows **rlstack** to ignore layers upon layers of abstraction,
  greatly simplifying the overall library implementation.

The end result is a high throughput library that can train policies to solve
complex tasks on a single, off-the-shelf computing device within minutes.

Unfortunately, this means **rlstack** doesn't support as many use cases as
a monolithic RL library might. In fact, **rlstack** is probably a bad fit for
your use case if:

* Your environment isn't parallelizable.
* Your environment must contain terminal conditions and can't be reformulated
  as an infinite horizon task.
* You want to use an ML framework that isn't PyTorch or you want to use an
  algorithm that isn't a variant of PPO.

However, if **rlstack** does fit your use case, it can do wonders for your
RL workflow.

Related Projects
================

* `RL Games`_: RL Games is a high performance RL library built around popular
  environment protocols.
* `RLlib`_: Ray's RLlib is the industry standard RL library that supports many
  popular RL algorithms. RLlib can scale RL workloads from your laptop all the
  way to the cloud with little-to-no changes to your code.
* `Sample Factory`_: Sample Factory provides an efficient and high quality
  implementation of PPO with a focus on accelerating training for a single machine
  with support for a wide variety of environment protocols.
* `SKRL`_: SKRL focuses on readability, simplicity, and transparency of RL algorithm
  implementations with support for a wide variety of environment protocols.
* `Stable Baselines 3`_: Stable Baselines 3 is a set of reliable and user-friendly
  RL algorithm implementations that integrate with a rich set of features desirable
  by most practitioners and use cases.
* `TorchRL`_: TorchRL is PyTorch's RL library that's focused on efficient, modular,
  documented, and tested RL building blocks and algorithm implementations aimed
  at supporting research in RL.

.. _`RL Games`: https://github.com/Denys88/rl_games
.. _`RLlib`: https://docs.ray.io/en/latest/rllib/index.html
.. _`Sample Factory`: https://github.com/alex-petrenko/sample-factory
.. _`SKRL`: https://github.com/Toni-SM/skrl
.. _`Stable Baselines 3`: https://github.com/DLR-RM/stable-baselines3
.. _`TorchRL`: https://github.com/pytorch/rl
