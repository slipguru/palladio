.. _framework:

The framework
=============

A dataset consists of two things:

* An input matrix :math:`X \in \mathbb{R}^{n \times p}` representing :math:`n` samples each one described by :math:`p` features; in the case of gene expression microarrays for instance each feature represents
* An output vector :math:`{\bf y}` of length :math:`n` whose elements are either a continuous value or a discrete label, describing some property of the samples. These may represent for example the levels of a given substance in the blood of an individual (continuous variable) or the *class* to which he or she belongs (for instance, someone affected by a given disease or a healthy control).

For the time being, we will only consider a specific instance of the latter case, where the number of classes is two: this is commonly referred to as *binary classification* scenario.

As previously explained, the core idea behind **PALLADIO** is to return, together with a list of significant features, not just a single value as an estimate for the prediction accuracy which can be achieved, but a distribution, so that it can be compared with the distribution obtained from experiments when the function is learned from data where the labels have been randomly shuffled (see :ref:`introduction`).

.. _pipeline:

Pipeline
----------------

Once the main script has been launched, the configuration file is read in order to retrieve all required information to run the experiments. These include:

* The location of **data** and **labels** files.
* Experiment design parameters, such as the total number of experiments and the ratio of samples to be used for testing in each experiment.
* Parameters specific to the chosen machine learning algorithm: for instance, for the :math:`\ell_1 \ell_2` regularized algorithm, the range for the :math:`\tau` and :math:`\lambda` parameters.

Then, experiments are distributed among the machines of the cluster; each machine will be assigned roughly the same number of jobs in order to balance the load.

Experiments
^^^^^^^^^^^^

Each experiment is divided in several stages, as shown in :numref:`experiment-stages`:

.. figure:: experiment_stages.pdf
   :scale: 80 %
   :align: center
   :alt: broken link
   :name: experiment-stages

   The stages each experiment goes through.


Dataset split and preprocessing
""""""""""""""""""""""""""""""""

Model selection
""""""""""""""""

Model assessment
""""""""""""""""

Analysis
-----------------
