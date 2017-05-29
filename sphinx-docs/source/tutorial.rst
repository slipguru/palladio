.. _tutorial:

Quick start tutorial
====================
**PALLADIO** may be installed using standard Python tools (with
administrative or sudo permissions on GNU-Linux platforms)::

    $ pip install palladio

    or

    $ easy_install palladio

We strongly suggest to use `Anaconda <https://www.continuum.io/downloads>`_ and create an environment for your experiments.

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
source package which contains a ``palladio/config_templates`` directory with some data files (``.npy`` or ``.csv``) which will be used to show **PALLADIO** functionalities.

**PALLADIO** needs only 3 ingredients:

* ``n_samples x n_variables`` input matrix
* ``n_samples x 1`` labels vector
* a ``configuration`` file

.. _cluster-setup:

Cluster setup
-----------------------

Since all experiments performed during a run are independent from one another, **PALLADIO** has been designed specifically to work in a cluster environment.
It is fairly easy to prepare the cluster for the experiments: assuming a standard configuration for the nodes (a shared home folder and a python installation which includes standard libraries for scientific computation, namely ``numpy``, ``scipy`` and ``sklearn``, as well as of course the ``mpi4py`` library for the MPI infrastructure), it is sufficient to transfer on the cluster a folder containing the dataset (data matrix and labels) and the configuration file, and install **PALLADIO** itself following the instructions above.
 .. and all additional libraries required by **PALLADIO** (``l1l2py``, available `here <http://slipguru.disi.unige.it/Software/L1L2Py/>`_ [#f1]_), together with **PALLADIO** itself of course (copying the package folders should be enough).

.. The content of the home folder once all required objects have been transfered to the cluster should look like this::
..
..     $ ls
..     pd_run.py experiment_folder l1l2py palladio
..
..     $ ls experiment_folder
..     data_file.csv labels_file.csv config.py

.. .. _quick-deployment:
..
.. Quick deployment
.. ^^^^^^^^^^^^^^^^
..
.. Moreover, a script is provided to speed up the deployment process. Simply run::
..
..     $ python palladio-release-folder/scripts/deploy.py [--sample-data] [DESTINATION_FOLDER]
..
.. This script will automatically copy all required files and libraries in the user home folder or ``DESTINATION_FOLDER`` if specified. The ``--sample-data`` option also copies a sample dataset in the home folder, to check if the installation was successful.

.. .. _input-data-format:

.. Input data format
.. -----------------
.. Input data (gene expression matrix and labels) are assumed to be textual ad
.. separated by a char (delimiter).
.. For example, the given data matrix (of Leukemia gene expressions) is a text file
.. where samples are organized by columns and microarray probes by row and gene
.. expressions values are separated by a comma (``','``).
..
.. .. literalinclude:: ./gedm_trunc.csv
..    :lines: 1, 150-160
..    :append: ...
..
.. Labels contains information about the given samples, indicated if they belong
.. to the ALL (Acute Lymphoblastic Leukemia) or AML (Acute Myeloid Leukemia) group:
..
.. .. literalinclude:: ../../example/data/labels.csv
..    :lines: 1-6, 29-34
..    :append: ...

.. _configuration:

Configuration File
------------------
**PALLADIO** configuration file is a standard Python script. It is
imported as a module, then all the code is executed. In this file the user defines all the parameters required to run a *session*, that is to perform all the experiments required to produce the final plots and reports.

In folder ``palladio/config_templates`` you will find an example of a typical configuration file.
Every configuration file has several sections which control different aspects of the procedure.

The code below contains all the information required to load the dataset which will be used in the experiments.

.. code-block:: python

  data_path = 'data/gedm.csv'
  target_path = 'data/labels.csv'

  # pandas.read_csv options
  data_loading_options = {
      'delimiter': ',',
      'header': 0,
      'index_col': 0
  }
  target_loading_options = data_loading_options

  dataset = datasets.load_csv(
      os.path.join(os.path.dirname(__file__),data_path),
      os.path.join(os.path.dirname(__file__),target_path),
      data_loading_options=data_loading_options,
      target_loading_options=target_loading_options,
      samples_on='col')

  data, labels = dataset.data, dataset.target
  feature_names = dataset.feature_names

The last two lines store the input data matrix ``data`` and the labels vector ``labels`` in two variables which will be accessible during the session. The names of the features are also saved at this point.
Notice how it is possible to load the dataset in any desired way, as long as ``data`` ends up being a :math:`n \times d` matrix and ``labels`` a vector of :math:`n` elements (both ``np.array``-like).

Next, we have the section containing settings relative to the session itself:

.. code-block:: python

  session_folder = 'palladio_test_session'

  # The learning task, if None palladio tries to guess it
  # [see sklearn.utils.multiclass.type_of_target]
  learning_task = None

  # The number of repetitions of 'regular' experiments
  n_splits_regular = 50

  # The number of repetitions of 'permutation' experiments
  n_splits_permutation = 50

The most important settings are the last two, namely ``n_splits_regular`` and ``n_splits_permutation``, which control how many repetitions of *regular* and *permutations* experiments are performed.
Normally you'll want to perform the same number of experiments for the two *batches*, but there are cases in which for instance you may want to perform only one of the two batches: in that case you will want to set one of the two variables to be 0.

Finally, the section of the configuration file where the actual variable selection and learing algorithms (and their parameters) are chosen:

.. code-block:: python

  model = RFE(LinearSVC(loss='hinge'), step=0.3)

  # Set the estimator to be a GridSearchCV
  param_grid = {
      'n_features_to_select': [10, 20, 50],
      'estimator__C': np.logspace(-4, 0, 5),
  }

  estimator = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1)

  # Set options for ModelAssessment
  ma_options = {
      'test_size': 0.25,
      'scoring': 'accuracy',
      'n_jobs': -1,
      'n_splits': n_splits_regular
  }

This is maybe the less intuitive part of the file.
Because of the way **PALLADIO** is designed, for all repetitions of the experiment a new learning and test set are generated by resampling without replacement from the whole dataset, then an estimator is used to fit the learning set. This is where that estimator (and its parameter) is defined.

Think about the ``estimator`` variable as the ``sklearn``-compatible object (an estimator) which you would use to fit a training set, with the intent of validating it on a separate test set.

In this example we use a RFE algorithm (see `<http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html>`_) for variable selection, which internally uses a Linear SVM for classification.
Then we use a ``GridSearchCV`` (`<http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_) object to wrap the RFE object, because we want to optimize the parameters for the RFE object itself, which are defined just above the declaration of the ``estimator`` variable.

The dictionary ``ma_options`` define some more configuration options for the :py:class:`~palladio.model_assessment.ModelAssessment` object, which is the one responsible for the outer iterations (the ones where the dataset is resampled);
the ``test_size`` key for instance determins the portion of data left aside for testing.


.. _running-experiments:

Running the experiments
-----------------------

Parallel jobs are created by invoking the ``mpirun`` command; the following syntax assumes that the `OpenMPI <https://www.open-mpi.org/>`_ implementation of MPI has been chosen for the cluster, if this is not the case, please refer to the documentation of the implementation available on your cluster for the command line options corresponding to those specified here::

    $ mpirun -np N_JOBS --hostfile HOSTFILE pd_run.py path/to/config.py

Here ``N_JOBS`` obviously determines how many parallel jobs will be spawned and distributed among all available nodes, while ``HOSTFILE`` is a file listing the addresses or names of the available nodes.

Take into account that if optimized linear algebra libraries are present on the nodes (as it is safe to assume for most clusters) you should tune the number of jobs so that cores are optimally exploited: since those libraries already parallelize operations, it is useless to assign too many slots for each node.

Running experiments on a single machine
"""""""""""""""""""""""""""""""""""""""""

It is possible to perform experiments using **PALLADIO** also on a single machine, without a cluster infrastructure. The command is similar to the previous one, it is sufficient to omit the first part, relative to the MPI infrastructure::

    $ pd_run.py path/to/config.py

.. warning::

  Due to the great number of experiments which are performed, it might take a very long time for the whole procedure to complete; this option is therefore deprecated unless the dataset is very small (no more than 100 samples and no more than 100 features).

.. _results-analysis:

Results analysis
----------------
The ``pd_analysis.py`` script reads the results from all experiments and produces several plots and text files. The syntax is the following::

    $ pd_analysis.py path/to/results_dir

See :ref:`analysis` for further details on the output of the analysis.


.. Reference
.. ----------------
.. .. [1] Weinstein, John N., et al. "The cancer genome atlas pan-cancer analysis project." Nature genetics 45.10 (2013): 1113-1120.


.. https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test

.. .. rubric:: Footnotes

.. .. [#f1] A standalone version of the ``L1L2Py`` library is included in the package in order to further speed up the deployment process.
