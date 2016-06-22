.. _introduction:

Introduction
============

The issue of reproducibility of experiments is of paramount importance in scientific studies, as it influences the reliability of published findings. However when dealing with biological data, especially genomic data such as gene expression or SNP microarrays, it is not uncommon to have a very limited number of samples available, represented by a huge number of *features*.

A dataset often consists of two things:

* An input matrix :math:`X \in \mathbb{R}^{n \times p}` representing :math:`n` samples each one described by :math:`p` features; in the case of gene expression microarrays for instance each feature represents
* An output vector :math:`{\bf y}` of length :math:`n` whose elements are either a continuous value or a discrete label, describing some property of the samples. These may represent for example the levels of a given substance in the blood of an individual (continuous variable) or the *class* to which he or she belongs (for instance, someone affected by a given disease or a healthy control).
We will limit for the time being for a specific instance of the latter case, where the number of classes is two: this is a *binary classification* scenario.

Using Machine Learning for the analysis of data means using part of the dataset (called *training set*) to fit a model, i.e. a function which takes an element from the input matrix and *predicts* the corresponding output value (for the binary classification case, the class to which that sample belongs).
