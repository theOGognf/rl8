rlstack: A Minimal RL Library
=============================

**rlstack** is a minimal RL library that can simulate highly parallelized,
infinite horizon environments, and can train a PPO policy using those
environments, achieving up to 500k environment transitions (and one policy
update) per second using a single NVIDIA RTX 2080.

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
high-level trainer interface (this updates the policy indefinitely).

.. code:: python

    from rlstack import Trainer
    from rlstack.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv)
    trainer.run()

Concepts
========

**rlstack** is minimal in that it limits the number of interfaces required for
training a policy with PPO, even for customized policies, without restrictions
on observation and action specs, custom models, and custom action
distributions.

**rlstack** is built around six key concepts:

* **The environment**: The simulation that the policy learns to interact with.
  The environment is *always user-defined*.
* **The model**: The policy parameterization that determines how the policy
  processes environment observations and how parameters for the action
  distribution are generated. The model is *usually user-defined*
  (default models are sometimes sufficient depending on the environment's
  observation and action specs).
* **The action distribution**: The mechanism for representing actions
  conditioned on environment observations and model outputs. Environment
  actions are ultimately sampled from the action distribution.
  The action distribution is *sometimes user-defined* (default action
  distributions are usually sufficient depending on the environment's
  observation and action specs).
* **The policy**: The union of the model and the action distribution that
  actually calls and samples from the model and action distribution,
  respectively. The policy handles some pre/post -processing on its I/O
  to make it more convenient to sample from the model and action distribution
  together. The policy is *almost never user-defined*.
* **The algorithm**: The PPO implementation that uses the environment to train
  the policy (i.e., update the model's parameters). All hyperparameters and
  customizations are set with the algorithm. The algorithm is *almost never
  user-defined*.
* **The trainer**: The high-level interface for using the algorithm to train
  indefinitely or until some condition is met. The trainer directly integrates
  with MLFlow to track experiments and training progress. The trainer is *never
  user-defined*.

Quick Examples
==============

Customizing Training Runs
-------------------------

Use a custom distribution and custom hyperparameters with the low-level
algorithm interface. The algorithm uses default feedforward models depending
on the environment's action spec.

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

Specify the same settings using the high-level trainer interface.

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
depending on the environment's action spec.

.. code:: python

    from rlstack import RecurrentAlgorithm
    from rlstack.env import DiscreteDummyEnv

    algo = RecurrentAlgorithm(DiscreteDummyEnv)
    algo.collect()
    algo.step()

Specify the algorithm type using the high-level trainer interface (which
usually defaults to a feedforward version of the algorithm).

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

Specify training on a GPU using the high-level trainer interface.

.. code:: python

    from rlstack import Trainer
    from rlstack.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv, algorithm_config={"device": "cuda"})
    trainer.run()

Minimizing GPU Memory Usage
---------------------------

Use the low-level algorithm interface to enable policy updates with gradient
accumulation and/or `Automatic Mixed Precision (AMP)`_ to minimize GPU memory
usage so you can simulate more environments or use larger models.

.. code:: python

    import torch.optim as optim

    from rlstack import Algorithm
    from rlstack.env import DiscreteDummyEnv

    algo = Algorithm(
        DiscreteDummyEnv,
        optimizer_cls=optim.SGD,
        accumulate_grads=True,
        enable_amp=True,
        sgd_minibatch_size=8192,
        device="cuda",
    )
    algo.collect()
    algo.step()

Enable memory-minimization settings using the high-level trainer interface.

.. code:: python

    import torch.optim as optim

    from rlstack import Trainer
    from rlstack.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv,
        algorithm_config={
            "optimizer_cls": optim.SGD,
            "accumulate_grads": True,
            "enable_amp": True,
            "sgd_minibatch_size": 8192,
            "device": "cuda",
        }
    )
    trainer.run()

Specifying Training Stop Conditions
-----------------------------------

Specify training stop conditions based on training statistics using the
high-level trainer interface.

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

**TL;DR: rlstack focuses on a niche subset of RL that simplifies the overall
library while allowing fast and fully customizable environments, models, and
action distributions.**

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

The end result is a minimal and high throughput library that can train policies
to solve complex tasks on a single NVIDIA RTX 2080 within minutes.

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

.. _`Automatic Mixed Precision (AMP)`: https://pytorch.org/docs/stable/amp.html
.. _`RL Games`: https://github.com/Denys88/rl_games
.. _`RLlib`: https://docs.ray.io/en/latest/rllib/index.html
.. _`Sample Factory`: https://github.com/alex-petrenko/sample-factory
.. _`SKRL`: https://github.com/Toni-SM/skrl
.. _`Stable Baselines 3`: https://github.com/DLR-RM/stable-baselines3
.. _`TorchRL`: https://github.com/pytorch/rl
