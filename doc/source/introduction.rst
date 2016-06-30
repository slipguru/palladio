.. _introduction:

Introduction
============

The issue of reproducibility of experiments is of paramount importance in scientific studies, as it influences the reliability of published findings. However when dealing with biological data, especially genomic data such as gene expression or SNP microarrays, it is not uncommon to have a very limited number of samples available, and these are usually represented by a huge number of measurements.

A common scenario is the so called *case-control study*: some quantities (e.g., gene expression levels, presence of alterations in key *loci* in the genome) are measured in a number of individuals who may be divided in two groups, or classes, depending whether they are affected by some kind of disease or not; the goal of the study is to find **which ones**, if any, among the possibly many measurements, or *features*, taken from the individuals (*samples*), can be used to define a *function* able to *predict*, to some extent, to which *class* (in this case, a diseased individual or a healthy one) an individual belongs.

Machine Learning (ML) techniques work by *learning* such function using only *part* of the available samples (the *training set*), so that the remaining ones (the *test set*) can be used to determine how well the function is able to predict the class of **new** samples; this is done, roughly speaking, to ensure that the function is able to capture some real characteristics of the data and not simply fitting the training data, which is trivial.
This is referred to in ML literature as *binary classification scenario*.

In the aforementioned scenario, having only few samples available means that the learned function may be highly dependent on how the dataset was split; a common solution to this issue is to perform *K-fold cross validation* (KCV) which means splitting the dataset in :math:`K` chunks and performing the experiment :math:`K` times, each time leaving out a different chunk to be used as test set; this reduces the risk that the results are dependent on a particular split. The :math:`K` parameter usually is chosen between 3 and 10, depending on the dataset.

This is the idea behind `L1L2Signature <http://slipguru.disi.unige.it/Software/L1L2Signature/>`_ , a framework specifically designed with this issue in mind.
``L1L2Signature`` performs *feature selection* while learning the function, that is it tries to identify which ones among the available features are actually *relevant* for the problem, that is *which are actually used* in the learned function.
The output of ``L1L2Signature`` consists of a *signature*, that is a list of relevant features, as well as a measure of *prediction accuracy*, that is the ratio of correctly classified samples in the test set, averaged over all splits.

There are however cases where it is hard to tell whether this procedure actually yielded a meaningful result: for instance, the fact that the accuracy measure is only *slightly* higher than chance can indicate two very different things:

* The available features can only describe the phenomenon to a limited extent.
* There is actually no relationship between features and output class, and getting a result better than chance was just a matter of luck in the subdivision of the dataset.

In order to tackle this issue, **PALLADIO** repeats the experiment many times (:math:`\sim 100`), each time using a different training and test set by randomly sampling from the whole original dataset (without replacement).
The experiment is also repeated the same number of times in a similar setting with a difference: in training sets, the labels are randomly shuffled, therefore destroying any connection between features and output class.

The output of this procedure is not a single value, possibly averaged, for the accuracy, but instead *two distributions of values* (one for each of the two settings described above) which, in case of datasets where the relationship between features and output class is at most faint, allows users to distinguish between the two scenarios mentioned above: in facts, if the available features are somehow connected with the outcome class, even weakly, then the two distributions will be  different enough to be distinguished; if on the other hand features and class are not related in any way, the two distributions will be indistinguishable, and it will be safe to draw that conclusion.
