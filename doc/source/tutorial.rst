.. _tutorial:

Quick start tutorial
====================
**PALLADIO** may be installed using standard Python tools (with
administrative or sudo permissions on GNU-Linux platforms)::

    $ pip install palladio

    or

    $ easy_install palladio

Installation from sources
-------------------------
If you like to manually install **PALLADIO**, download the .zip or .tar.gz archive
from `<http://slipguru.github.io/palladio/>`_. Then extract it and move into the root directory::

    $ unzip slipguru-palladio-|release|.zip
    $ cd palladio-|release|/

or::

    $ tar xvf slipguru-palladio-|release|.tar.gz
    $ cd palladio-|release|/

Otherwise you can clone our `GitHub repository <https://github.com/slipguru/palladio>`_::

   $ git clone https://github.com/slipguru/palladio.git

From here, you can follow the standard Python installation step::

    $ python setup.py install

After **PALLADIO** installation, you should have access to two scripts,
named with a common ``pd_`` prefix::

    $ pd_<TAB>
    pd_analysis.py    pd_run.py

This tutorial assumes that you downloaded and extracted **PALLADIO**
source package which contains a ``examples\data`` directory with some data files (``.npy`` or ``.csv``) which will be used to show **PALLADIO** functionalities.

**PALLADIO** needs only 3 ingredients:

* ``n_samples x n_variables`` input matrix
* ``n_samples x 1`` labels vector
* ``configuration`` file

.. _input-data-format:

Input data format
-----------------
Input data (gene expression matrix and labels) are assumed to be textual ad
separated by a char (delimiter).
For example, the given data matrix (of Leukemia gene expressions) is a text file
where samples are organized by columns and microarray probes by row and gene
expressions values are separated by a comma (``','``).

.. literalinclude:: ../../example/data/gedm.csv
   :lines: 1, 150-160
   :append: ...

Labels contains information about the given samples, indicated if they belong
to the ALL (Acute Lymphoblastic Leukemia) or AML (Acute Myeloid Leukemia) group:

.. literalinclude:: ../../example/data/labels.csv
   :lines: 1-6, 29-34
   :append: ...

.. _configuration:

Configuration File
------------------
**PALLADIO** configuration file is a standard Python script. It is
imported as a module, then all the code is executed. In this file the user can define all the options needed to setup the experiment.

.. literalinclude:: ../../example/config_l1l2.py
   :language: python

.. _cluster-setup:

Cluster setup
-----------------------

Since all experiments performed during a run are independent from one another, **PALLADIO** has been designed specifically to work in a cluster environment.
It is fairly easy to prepare the cluster for the experiments: assuming a standard configuration for the nodes (a shared home folder and a python installation which includes standard libraries for scientific computation, namely ``numpy``, ``scipy`` and ``sklearn``, as well as of course the ``mpi4py`` library for the MPI infrastructure), it is sufficient to transfer on the cluster a folder containing the dataset (data matrix and labels) and the configuration file and all additional libraries required by **PALLADIO** (``l1l2py``, available `here <http://slipguru.disi.unige.it/Software/L1L2Py/>`_), together with **PALLADIO** itself of course (copying the package folders should be enough).

The content of the home folder once all required objects have been transfered to the cluster should look like this::

    $ ls
    pd_run.py experiment_folder l1l2py palladio

    $ ls experiment_folder
    data_file.csv labels_file.csv config.py

.. _running-experiments:

Running the experiments
-----------------------

Parallel jobs are created by invoking the ``mpirun`` command; the following syntax assumes that the `OpenMPI <https://www.open-mpi.org/>`_ implementation of MPI has been chosen for the cluster, if this is not the case, please refer to the documentation of the implementation available on your cluster for the command line options corresponding to those specified here::

    $ mpirun -np N_JOBS --hostfile HOSTFILE python pd_run.py path/to/config.py

Here ``N_JOBS`` obviously determines how many parallel jobs will be spawned and distributed among all available nodes, while ``HOSTFILE`` is a file listing the addresses or names of the available nodes.

Take into account that if optimized linear algebra libraries are present on the nodes (as it is safe to assume for most clusters) you should tune the number of jobs so that cores are optimally exploited: since those libraries already parallelize operations, it is useless to assign too many slots for each node.

.. _results-analysis:

Results analysis
----------------
The ``pd_analysis.py`` script reads the results from all experiments and produces several plots and text files. The syntax is the following::

    $ pd_analysis.py path/to/results_dir

:numref:`manhattan-plot` shows the absolute feature selection frequency in both *regular* experiments and permutation tests; each tick on the horizontal axis represents a different feature, whose position on the vertical axis is the number of times it was selected in an experiment. Features are sorted based on the selection frequency relative to *regular* experiments; green dots are frequencies for *regular* experiments, red ones for permutation tests.

.. figure:: manhattan_plot.pdf
   :scale: 80 %
   :align: center
   :alt: broken link
   :name: manhattan-plot

   A manhattan plot showing the distribution of frequencies for both *regular* experiments and permutation tests.

:numref:`signature-frequencies` shows a detail of the frequeny of the top :math:`2 \times p_{\rm rel}` selected features, where :math:`p_{\rm rel}` is the number of features identified as *relevant* by the framework, i.e. those which have been selected enough times according to the selection threshold defined in the configuration file. Seeing the selection frequency of *relevant* features with respect to the selection frequency of those which have been rejected may help better interpret the obtained results.

.. figure:: signature_frequencies.pdf
  :scale: 80 %
  :align: center
  :alt: broken link
  :name: signature-frequencies

  A detail of the manhattan plot.

Finally, :numref:`permutation-acc-distribution` shows the distribution of prediction accuracies (corrected for class imbalance) for *regular* experiments and permutation tests; this plot answer the questions:

* Is there any signal in the data being analyzed?
* If yes, how much the model can describe it?

In the example figure, the two distributions are clearly different, and the green one (showing the accuracies of *regular* experiments) has a mean which is significantly higher than chance (50 \%). A p-value obtained with the Wilcoxon rank sum test is also present in this plot, indicating whether there is a significant difference between the two distributions.

.. figure:: permutation_acc_distribution.pdf
  :scale: 80 %
  :align: center
  :alt: broken link
  :name: permutation-acc-distribution

  The distributions of accuracies for both *regular* experiments and permutation tests.

.. Reference
.. ----------------
.. .. [1] Weinstein, John N., et al. "The cancer genome atlas pan-cancer analysis project." Nature genetics 45.10 (2013): 1113-1120.


.. https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
