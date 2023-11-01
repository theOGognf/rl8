rl8: A Minimal End-to-End RL Library
====================================

**rl8** is a minimal end-to-end RL library that can simulate highly
parallelized, infinite horizon environments, and can train a PPO policy
using those environments, achieving up to 1M environment transitions
(and one policy update) per second using a single NVIDIA RTX 2080.

* **Documentation**: https://theogognf.github.io/rl8/
* **PyPI**: https://pypi.org/project/rl8/
* **Repository**: https://github.com/theOGognf/rl8

Quick Start
===========

Installation
------------

Install with pip for the latest (stable) version.

.. code:: console

    pip install rl8

Install from GitHub for the latest (unstable) version.

.. code:: console

    git clone https://github.com/theOGognf/rl8.git
    pip install ./rl8/

Basic Usage
-----------

Train a policy with PPO and log training progress with MLflow using the
high-level trainer interface (this updates the policy indefinitely).

.. code:: python

    from rl8 import Trainer
    from rl8.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv)
    trainer.run()

Collect environment transitions and update a policy directly using the
low-level algorithm interface (this updates the policy once).

.. code:: python

    from rl8 import Algorithm
    from rl8.env import DiscreteDummyEnv

    algo = Algorithm(DiscreteDummyEnv)
    algo.collect()
    algo.step()

The trainer interface is the most popular interface for policy training
workflows, whereas the algorithm interface is useful for lower-level
customization of policy training workflows.

Concepts
========

**rl8** is minimal in that it limits the number of interfaces required for
training a policy with PPO, even for customized policies, without restrictions
on observation and action specs, custom models, and custom action
distributions.

**rl8** is built around six key concepts:

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
  together. The policy is *rarely user-defined*.
* **The algorithm**: The PPO implementation that uses the environment to train
  the policy (i.e., update the model's parameters). All hyperparameters and
  customizations are set with the algorithm. The algorithm is *rarely
  user-defined*.
* **The trainer**: The high-level interface for using the algorithm to train
  indefinitely or until some condition is met. The trainer directly integrates
  with MLflow to track experiments and training progress. The trainer is *rarely
  user-defined*.

Quick Examples
==============

Customizing Training Runs
-------------------------

Use a custom distribution and custom hyperparameters by passing
options to the trainer (or algorithm) interface.

.. code:: python

    from rl8 import SquashedNormal, Trainer
    from rl8.env import ContinuousDummyEnv

    trainer = Trainer(
        ContinuousDummyEnv,
        distribution_cls=SquashedNormal,
        gae_lambda=0.99,
        gamma=0.99,
    )
    trainer.run()

Training a Recurrent Policy
---------------------------

Swap to the recurrent flavor of the trainer (or algorithm) interface
to train a recurrent model and policy. The recurrent interfaces use
canned and default recurrent models depending on the environment's
observation and action specs.

.. code:: python

    from rl8 import RecurrentTrainer
    from rl8.env import DiscreteDummyEnv

    trainer = RecurrentTrainer(DiscreteDummyEnv)
    trainer.run()

Training on a GPU
-----------------

Specify the device used across the environment, model, and
algorithm.

.. code:: python

    from rl8 import Trainer
    from rl8.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv, device="cuda")
    trainer.run()

Minimizing GPU Memory Usage
---------------------------

Enable policy updates with gradient accumulation and/or
`Automatic Mixed Precision (AMP)`_ to minimize GPU memory
usage so you can simulate more environments or use larger models.

.. code:: python

    import torch.optim as optim

    from rl8 import Trainer
    from rl8.env import DiscreteDummyEnv

    trainer = Trainer(
        DiscreteDummyEnv,
        optimizer_cls=optim.SGD,
        accumulate_grads=True,
        enable_amp=True,
        sgd_minibatch_size=8192,
        device="cuda",
    )
    trainer.run()

Specifying Training Stop Conditions
-----------------------------------

Specify training stop conditions based on training statistics to stop
training early when statistics plateau, hit a limit, stop
increasing or decreasing, etc..

.. code:: python

    from rl8 import Trainer
    from rl8.conditions import Plateaus
    from rl8.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv)
    trainer.run(stop_conditions=[Plateaus("returns/mean", rtol=0.05)])

Why rl8?
============

**TL;DR: rl8 focuses on a niche subset of RL that simplifies the overall
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

**rl8** is a niche RL library that finds a goldilocks zone between the
feature support and speed/complexity tradeoff by making some key assumptions:

* Environments are highly parallelized and their parallelization is entirely
  managed within the environment. This allows **rl8** to ignore distributed
  computing design considerations.
* Environments are infinite horizon (i.e., they have no terminal conditions).
  This allows **rl8** to reset environments at the same, fixed horizon
  intervals, greatly simplifying environment and algorithm implementations.
* The only supported ML framework is PyTorch and the only supported algorithm
  is PPO. This allows **rl8** to ignore layers upon layers of abstraction,
  greatly simplifying the overall library implementation.

The end result is a minimal and high throughput library that can train policies
to solve complex tasks on a single NVIDIA RTX 2080 within minutes.

Unfortunately, this means **rl8** doesn't support as many use cases as
a monolithic RL library might. In fact, **rl8** is probably a bad fit for
your use case if:

* Your environment isn't parallelizable.
* Your environment must contain terminal conditions and can't be reformulated
  as an infinite horizon task.
* You want to use an ML framework that isn't PyTorch or you want to use an
  algorithm that isn't a variant of PPO.

However, if **rl8** does fit your use case, it can do wonders for your
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
  at supporting research in RL. TorchRL is a direct dependency of **rl8**.

.. _`Automatic Mixed Precision (AMP)`: https://pytorch.org/docs/stable/amp.html
.. _`RL Games`: https://github.com/Denys88/rl_games
.. _`RLlib`: https://docs.ray.io/en/latest/rllib/index.html
.. _`Sample Factory`: https://github.com/alex-petrenko/sample-factory
.. _`SKRL`: https://github.com/Toni-SM/skrl
.. _`Stable Baselines 3`: https://github.com/DLR-RM/stable-baselines3
.. _`TorchRL`: https://github.com/pytorch/rl
