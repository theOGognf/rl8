Pendulum
========

A simple reimplementation of the classic `Pendulum`_ environment.

The system consists of a pendulum attached at one end to a fixed point,
and the other end being free. The pendulum starts in a random position
and the goal is to apply torque on the free end to swing it into an
upright position, with its center of gravity right above the fixed point.

Organization
------------

It's recommended that you browse the example's files to gain an understanding
of how custom environments and models should be defined to get the most
benefit out of **rl8**'s design. The example is organized as follows:

* ``./__main__.py`` is the main script.
* ``./env.py`` contains the environment's definition.

.. _`MountainCar`: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py
