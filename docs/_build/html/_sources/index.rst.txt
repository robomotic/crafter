OaK Framework Documentation
============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   strategy
   api/oak

Welcome to the OaK (Options and Knowledge) Framework documentation. This framework implements hierarchical reinforcement learning through reward-respecting subtasks, options, and general value functions.

Key Concepts
------------

- **SubTask**: Reward-respecting subtasks of feature attainment
- **Option**: Temporally extended policies with termination conditions
- **Model**: Models of option dynamics for planning
- **Planning**: High-level temporal abstraction planning using options

Core Modules
~~~~~~~~~~~~

.. autosummary::
   :toctree: api
   :recursive:

   oak.agent
   oak.gvf
   oak.options
   oak.subtasks
   oak.tech_tree
   oak.features
   oak.interrupts

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:

   ../oak/docs/STRATEGY.md

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
