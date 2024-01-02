CartPole
========

A simple reimplementation of the classic `CartPole`_ environment.

A pole is attached by an un-actuated joint to a cart, which moves
along a frictionless track. The pendulum is placed upright on the
cart and the goal is to balance the pole and maintain the cart's
position at the origin by applying forces in the left and right
directions on the cart.

In contrast to the classic CartPole environment, this implementation
uses a continuous reward to transform the environment into an infinite
horizon task.

Organization
------------

It's recommended that you browse the example's files to gain an understanding
of how custom environments and models should be defined to get the most
benefit out of **rl8**'s design. The example is organized as follows:

* ``./__main__.py`` is the main script.
* ``./env.py`` contains the environment's definition.

.. _`CartPole`: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
