MountainCar
===========

A simple reimplementation of the classic `MountainCar`_ environment.

A car is placed stochastically at the bottom of a sinusoidal valley,
with the only possible actions being accelerations that can be applied
to the car in either direction. The goal is to strategically accelerate
the car to reach a goal state on top of the right hill.

In contrast to the classic MountainCar environment, this implementation
has larger initial condition bounds, uses some reward shaping, and
doesn't terminate when the car is in a goal state.

Organization
------------

It's recommended that you browse the example's files to gain an understanding
of how custom environments and models should be defined to get the most
benefit out of **rl8**'s design. The example is organized as follows:

* ``./__main__.py`` is the main script.
* ``./env.py`` contains the environment's definition.

.. _`MountainCar`: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py
