# ![rl8 logo.][13]

![PyPI Downloads](https://img.shields.io/pypi/dm/rl8)
![PyPI Version](https://img.shields.io/pypi/v/rl8)
![Python Versions](https://img.shields.io/pypi/pyversions/rl8)

**rl8** is a minimal end-to-end RL library that can simulate highly
parallelized, infinite horizon environments, and can train a PPO policy
using those environments, achieving up to 1M environment transitions
(and one policy update) per second using a single NVIDIA RTX 2080.

* **Documentation**: https://theogognf.github.io/rl8/
* **PyPI**: https://pypi.org/project/rl8/
* **Repository**: https://github.com/theOGognf/rl8

The figure below depicts **rl8**'s experiment tracking integration with
[MLflow][3] and **rl8**'s ability to solve reinforcement learning problems
within seconds.

![Consistently solving CartPole within seconds.][12]

# Quick Start

## Installation

Install with pip for the latest stable version.

```console
pip install rl8
```

Install from GitHub for the latest unstable version.

```console
git clone https://github.com/theOGognf/rl8.git
pip install ./rl8/
```

## Basic Usage

Train a policy with PPO and log training progress with MLflow using the
high-level trainer interface (this updates the policy indefinitely).

```python
from rl8 import Trainer
from rl8.env import DiscreteDummyEnv

trainer = Trainer(DiscreteDummyEnv)
trainer.run()
```

Collect environment transitions and update a policy directly using the
low-level algorithm interface (this updates the policy once).

```python
from rl8 import Algorithm
from rl8.env import DiscreteDummyEnv

algo = Algorithm(DiscreteDummyEnv)
algo.collect()
algo.step()
```

The trainer interface is the most popular interface for policy training
workflows, whereas the algorithm interface is useful for lower-level
customization of policy training workflows.

# Concepts

**rl8** is minimal in that it limits the number of interfaces required for
training a policy with PPO without restrictions on observation and action
specs, custom models, and custom action distributions.

**rl8** is built around six key concepts:

* **[The environment][14]**: The simulation that the policy learns to interact with.
  The environment definition is a bit different from your typical environment
  definition from other RL libraries.
* **[The model][15]**: The policy parameterization that determines how the policy
  processes environment observations and how parameters for the action
  distribution are generated.
* **[The action distribution][16]**: The mechanism for representing actions
  conditioned on environment observations and model outputs. Environment
  actions are ultimately sampled from the action distribution.
* **[The policy][17]**: The union of the model and the action distribution that
  actually calls and samples from the model and action distribution,
  respectively.
* **[The algorithm][18]**: The PPO implementation that uses the environment to train
  the policy (i.e., update the model's parameters).
* **[The trainer][19]**: The high-level interface for using the algorithm to train
  indefinitely or until some condition is met. The trainer directly integrates
  with MLflow to track experiments and training progress.

# Quick Examples

These short snippets showcase **rl8**'s main features. See the [examples][2]
for complete implementations of **rl8**-compatible environments and models.

## Customizing Training Runs

Use a custom distribution and custom hyperparameters by passing
options to the trainer (or algorithm) interface.

```python
from rl8 import SquashedNormal, Trainer
from rl8.env import ContinuousDummyEnv

trainer = Trainer(
    ContinuousDummyEnv,
    distribution_cls=SquashedNormal,
    gae_lambda=0.99,
    gamma=0.99,
)
trainer.run()
```

## Training a Recurrent Policy

Swap to the recurrent flavor of the trainer (or algorithm) interface
to train a recurrent model and policy.

```python
from rl8 import RecurrentTrainer
from rl8.env import DiscreteDummyEnv

trainer = RecurrentTrainer(DiscreteDummyEnv)
trainer.run()
```

## Training on a GPU

Specify the device used across the environment, model, and
algorithm.

```python
from rl8 import Trainer
from rl8.env import DiscreteDummyEnv

trainer = Trainer(DiscreteDummyEnv, device="cuda")
trainer.run()
```

## Minimizing GPU Memory Usage

Enable policy updates with gradient accumulation and/or
[Automatic Mixed Precision (AMP)][1] to minimize GPU memory
usage so you can simulate more environments or use larger models.

```python
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
```

## Specifying Training Stop Conditions

Specify conditions based on training statistics to stop training early.

```python
from rl8 import Trainer
from rl8.conditions import Plateaus
from rl8.env import DiscreteDummyEnv

trainer = Trainer(DiscreteDummyEnv)
trainer.run(stop_conditions=[Plateaus("returns/mean", rtol=0.05)])
```

## Training with the CLI

Suppose `./config.yaml` contains the following.

```yaml
# Fully qualified path to the environment's definition.
env_cls: rl8.env.ContinuousDummyEnv

# Some custom parameters.
gamma: 0.75
horizon: 8
```

Train a policy with the trainer interface using the `rl8 train` CLI.

```console
rl8 train -f config.yaml
```

# Why rl8?

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

Rather than focusing on being a monolithic RL library, **rl8** fills the niche
of maximizing training performance for a few key assumptions:

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
to solve complex tasks within minutes on consumer grade compute devices.

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

# Related Projects

* [PureJaxRL][4]: PureJaxRL is a high-performance, end-to-end RL library. Think of
  it like **rl8**'s Jax equivalent, but more general in that it doesn't focus
  on infinite horizon tasks.
* [RL Games][5]: RL Games is a high performance RL library built around popular
  environment protocols.
* [RLlib][6]: Ray's RLlib is the industry standard RL library that supports many
  popular RL algorithms. RLlib can scale RL workloads from your laptop all the
  way to the cloud with little-to-no changes to your code.
* [Sample Factory][7]: Sample Factory provides an efficient and high quality
  implementation of PPO with a focus on accelerating training for a single machine
  with support for a wide variety of environment protocols.
* [SKRL][8]: SKRL focuses on readability, simplicity, and transparency of RL algorithm
  implementations with support for a wide variety of environment protocols.
* [Stable Baselines 3][9]: Stable Baselines 3 is a set of reliable and user-friendly
  RL algorithm implementations that integrate with a rich set of features desirable
  by most practitioners and use cases.
* [TorchRL][10]: TorchRL is PyTorch's RL library that's focused on efficient, modular,
  documented, and tested RL building blocks and algorithm implementations aimed
  at supporting research in RL. TorchRL is a direct dependency of **rl8**.
* [WarpDrive][11]: WarpDrive is a flexible, lightweight, and easy-to-use open-source
  RL framework that implements end-to-end multi-agent RL on a single or multiple
  GPUs. Think of it like **rl8**, but with an emphasis on support for multi-agent
  RL and without a focus on infinite horizon tasks.

[1]: https://pytorch.org/docs/stable/amp.html
[2]: https://github.com/theOGognf/rl8/tree/main/examples
[3]: https://github.com/mlflow/mlflow
[4]: https://github.com/luchris429/purejaxrl
[5]: https://github.com/Denys88/rl_games
[6]: https://docs.ray.io/en/latest/rllib/index.html
[7]: https://github.com/alex-petrenko/sample-factory
[8]: https://github.com/Toni-SM/skrl
[9]: https://github.com/DLR-RM/stable-baselines3
[10]: https://github.com/pytorch/rl
[11]: https://github.com/salesforce/warp-drive
[12]: https://raw.githubusercontent.com/theOGognf/rl8/main/docs/_static/rl8-examples-solving-cartpole.png
[13]: https://raw.githubusercontent.com/theOGognf/rl8/main/docs/_static/rl8-logo.png
[14]: https://theogognf.github.io/rl8/build/html/_modules/rl8/env.html#Env
[15]: https://theogognf.github.io/rl8/build/html/_modules/rl8/models/_base.html#GenericModelBase
[16]: https://theogognf.github.io/rl8/build/html/_modules/rl8/distributions.html#Distribution
[17]: https://theogognf.github.io/rl8/build/html/_modules/rl8/policies/_base.html#GenericPolicyBase
[18]: https://theogognf.github.io/rl8/build/html/_modules/rl8/algorithms/_base.html#GenericAlgorithmBase
[19]: https://theogognf.github.io/rl8/build/html/_modules/rl8/trainers/_base.html#GenericTrainerBase
