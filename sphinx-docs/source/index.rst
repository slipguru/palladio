.. palladio documentation master file

=================================================
PALLADIO (ParALleL frAmework for moDel selectIOn)
=================================================

**PALLADIO** [1]_ is a machine learning framework whose purpose is to provide robust and reproducible results when dealing with data where the signal to noise ratio is low.
It also provides tools to determine whether the dataset being analyzed contains any signal at all.
**PALLADIO** works by repeating the same experiment many times, each time resampling the learning and the test set so that the outcome is reliable as it is not determined by a *single* partition of the dataset. Besides, using permutation tests, it is possible to provide, to some extent, a measure of how reliable the results produced by an experiments are.
Since all experiments performed are independent, PALLADIO is designed so that it can exploit a cluster where it is available, in order to greatly reduce the amount of time required.

.. The final output of **PALLADIO** consists of:

The final output of **PALLADIO** consists of several plots and text reports. The main ones are:

* A plot showing the absolute frequencies of features for both *regular* experiments and permutation tests. Another plot shows in more detail the selection frequency for the most frequently selected features (i.e., those above the *selection threshold* defined in the configuration file).
* A plot showing the distribution of accuracies achieved by *regular* experiments and permutation tests.
* Two text files listing the features together with their absolute selection frequency, one for regular experiments and the other for permutation tests.

See :ref:`tutorial` for instructions on how to setup and launch a **PALLADIO** session.

.. only:: html

  A pdf version of this manual is available `here <./palladio.pdf>`_.

User documentation
==================

.. toctree::
  :maxdepth: 2

  introduction.rst

  framework.rst

  tutorial.rst

  api.rst


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


Reference
----------------
.. [1] Barbieri, M., Fiorini, S., Tomasi, F. and Barla, A. "PALLADIO: A Parallel Framework for Robust Variable Selection in High-dimensional Data." *Proceedings of the 6th Workshop on Python for High-Performance and Scientific Computing* (2016): 19-26.
