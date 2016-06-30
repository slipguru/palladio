.. palladio documentation master file

=================================================
PALLADIO (ParALleL frAmework for moDel selectIOn)
=================================================

**PALLADIO** is a machine learning framework whose purpose is to provide robust and reproducible results when dealing with data where the signal to noise ratio is low; it also provides tools to determine whether the dataset being analyzed contains any signal at all.
**PALLADIO** works by repeating the same experiment many times, each time resampling the training and the test set so that the outcome is reliable as it is not determined by a single partition of the dataset. Besides, using permutation tests, it is possible to provide, to some extent, a measure of how reliable the results produced by an experiments are.
Since all experiments performed are independent, PALLADIO is designed so that it can exploit a cluster where it is available, in order to greatly reduce the amount of time required for the experiment.

The final output of **PALLADIO** consists of:

* A plot showing the absolute frequencies of features for both *regular* experiments and permutation tests. Another plot shows in more detail the selection frequency for the most frequently selected features (i.e., those above the *selection threshold* defined in the configuration file).
* A plot showing the distribution of accuracies achieved by *regular* experiments and permutation tests.
* Two text files listing the features together with their absolute selection frequency, one for regular experiments and the other for permutation tests.


User documentation
==================
.. toctree::
   :maxdepth: 2

 introduction.rst

 framework.rst

 tutorial.rst

.. _api:

***********************
API
***********************

.. toctree::
   :maxdepth: 1


Pipeline utilities
-----------------------------

.. automodule:: palladio._core
   :members:

.. .. automodule:: adenine.core.pipelines
   :members:

.. .. automodule:: adenine.core.analyze_results
   :members:

.. Input Data
.. -----------------------------

.. .. automodule:: adenine.utils.data_source
   :members:


.. Plotting functions
.. -----------------------------

.. .. automodule:: adenine.core.plotting
   :members:


.. Extra tools
.. -----------------------------

.. .. automodule:: adenine.utils.extra
   :members:


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
