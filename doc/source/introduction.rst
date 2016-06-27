.. _introduction:

Introduction
============

The issue of reproducibility of experiments is of paramount importance in scientific studies, as it influences the reliability of published findings. However when dealing with biological data, especially genomic data such as gene expression or SNP microarrays, it is not uncommon to have a very limited number of samples available, and these are usually represented by a huge number of *features*.

Machine Learning (ML) techniques work by learning a model, i.e. some kind of *function* that is able to recognize patterns in the data, using only *part* of the available samples (the *training set*), so that the remaining ones (the *test set*) can be used to determine how well the model is able to describe the data. This is done, roughly speaking, to ensure that the function is able to capture some real characteristics of the data and not simply fitting the training data, which is trivial.

In the aforementioned scenario, having only few samples available means that the learned model may be highly dependent on how the dataset was split; a common solution to this issue is to perform *K-fold cross validation* (KCV) which means splitting the dataset in :math:`K` chunks and performing the experiment :math:`K` times, each time leaving out a different chunk to be used as test set; this reduces the risk that the results are dependent on a particular split. The :math:`K` parameter usually is chosen between 3 and 10, depending on the dataset.

This is the idea behind `L1L2Signature <http://slipguru.disi.unige.it/Software/L1L2Signature/>`_ , a framework specifically designed with this issue in mind. The output of ``L1L2Signature`` consists of a *signature*, that is a list of features considered *relevant* for the problem (as in, useful to describe the phenomenon being studied), and an estimate of how well the learned model would perform on new data.

There are however cases where it is hard to tell whether this procedure actually yielded a meaningful result: for instance, the fact that the accuracy measure is only *slightly* higher than chance can indicate two very different things:

* The available features can only describe the phenomenon to a limited extent.
* There is actually no pattern in the data, and getting a result better than chance was just a matter of luck in the subdivision of the dataset.

In order to tackle this issue, **PALLADIO** repeats the experiment many times (:math:`\sim 100`), each time using a different training and test set by randomly sampling from the whole original dataset (without replacement).
The experiment is also repeated the same number of times in a similar setting with a difference: in the training set, the data is processed so that any pattern which might have been present in the first place is destroyed.

The output of this procedure is not a single value, possibly averaged, for the accuracy, but instead *two distributions of values* (one for each of the two settings described above) which, in case of datasets where the pattern is not clearly visible, allows users to distinguish between the two aforementioned scenarios: in facts, if the available features are able to describe some kind of pattern in the data, even a small one, then the two distributions will be significantly different; if on the other hand *there simply is no pattern in the data*, the two distributions will be undistinguishable, and therefore it will be safe to conclude that there is no detectable pattern.


.. However that framework proves ineffective in cases where the pattern in the data is not well recognizable. This may happen for a number of reasons, such as the amount of noise or an insufficient number of available samples. In these cases it

.. Using Machine Learning for the analysis of data means using part of the dataset (called *training set*) to fit a model, i.e. a function which takes an element from the input matrix and *predicts* the corresponding output value (for the binary classification case, the class to which that sample belongs).


.. _framework:

The framework
=============

A dataset consists of two things:

* An input matrix :math:`X \in \mathbb{R}^{n \times p}` representing :math:`n` samples each one described by :math:`p` features; in the case of gene expression microarrays for instance each feature represents
* An output vector :math:`{\bf y}` of length :math:`n` whose elements are either a continuous value or a discrete label, describing some property of the samples. These may represent for example the levels of a given substance in the blood of an individual (continuous variable) or the *class* to which he or she belongs (for instance, someone affected by a given disease or a healthy control).
We will limit for the time being for a specific instance of the latter case, where the number of classes is two: this is commonly referred to as *binary classification* scenario.
