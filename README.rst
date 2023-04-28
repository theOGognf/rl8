rlstack: An RL Toolkit for the Sane
===================================

**rlstack** is a simple, high throughput, infinite horizon RL library that can
simulate highly parallelized environments, and can train a PPO policy using
those highly parallelized environments, achieving around 250k environment
transitions (and 1 update) per second using a single, off-the-shelf computing
device.

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

Train a policy with PPO and log training progress with MLFlow using the
high-level trainer interface.

.. code:: python

    from rlstack import Trainer
    from rlstack.env import DiscreteDummyEnv

    trainer = Trainer(DiscreteDummyEnv)
    trainer.run()

Collect environment transitions and update a policy directly using the
low-level algorithm interface.

.. code:: python

    from rlstack import Algorithm
    from rlstack.env import DiscreteDummyEnv

    algo = Algorithm(DiscreteDummyEnv)
    algo.collect()
    algo.step()

Related Projects
================

* ray.rllib: Ray's RLlib is the industry standard RL library. It supports all
  the popular RL algorithms and variants, and can scale workloads from your
  laptop all the way to the cloud.
* sample_factory: Sample Factory is similar to **rlstack** in that it provides
  an efficient and high quality implementation of PPO with a focus on accelerating
  training for a single machine.
* stable_baselines3: Stable Baselines 3 is a set of reliable and user-friendly
  RL algorithm implementations that integrate with a rich set of features desirable
  by most practitioners and use cases.

Why rlstack?
============

There are many high quality, open-sourced RL libraries. Most of them take on the
daunting task of being a monolithic, one-stop-shop for everything RL, attempting to
support as many algorithms, environments, models, and compute capabilities as possible.
Naturely, this monolothic goal has some drawbacks:

1) The software becomes more dense with each supported feature, making the library
    all-the-more difficult to customize for a specific use case.
2) The software becomes less performant for a specific use case, eventually
    requiring RL practitioners to wait weeks for experiment results, or to accept
    the cost of transitioning to expensive and difficult-to-manage compute
    clusters to get results faster.

There's a handful of high quality, open-sourced RL libraries that tradeoff feature
richness to reduce these drawbacks. However, each library still doesn't provide
enough speed benefit to warrant the switch from a monolithic repo, or is still
too complex to adapt to a specific use case.

**rlstack** is a niche and opinionated RL library that finds a goldilox zone
between the feature support and speed/complexity tradeoff by making some key
assumptions:

1) Environments are highly parallelized and their parallelization is entirely
   managed within the environment. This assumption allows **rlstack** to
   ignore distributed computing design considerations.
2) Environments are infinite horizon (i.e., they have no terminal conditions).
   This assumption allows **rlstack** to reset environments at the same,
   fixed horizon intervals, greatly simplifying environment and algorithm
   implementations.
3) The only supported ML framework is PyTorch and the only supported algorithm
   is PPO. This allows **rlstack** to ignore layers upon layers of abstraction,
   greatly simplifying the overall library implementation.

The end result is a simple and high throughput library that can train policies
to solve complex tasks on a single, off-the-shelf computing device within
minutes.

Unfortunately, this means **rlstack** doesn't support as many use cases as
a monolithic RL library might. In fact, **rlstack** is probably a bad fit for
your use case if:

1) Your environment isn't parallelizable.
2) Your environment must contain terminal conditions and can't be reformulated
   as an infinite horizon task.
2) You want to use an ML framework that isn't PyTorch or you want to use an
   algorithm that isn't a variant of PPO.

However, if **rlstack** does fit your use case, it can do wonders for your
RL workflow.
