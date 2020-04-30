# Alternating maximization: Unifying framework for 8 sparse PCA formulations and efficient parallel codes

Authors: [Peter Richtárik](https://richtarik.org/), [Majid Jahani](http://coral.ise.lehigh.edu/maj316/), [Selin Damla Ahipasaoglu](https://esd.sutd.edu.sg/people/faculty/selin-damla-ahipasaoglu) and [Martin Takáč](http://mtakac.com/)


## Introduction
Given a multivariate data set, sparse principal component analysis (SPCA) aims to extract several linear combinations of the variables that together explain the variance in the data as much as possible, while controlling the number of nonzero loadings in these combinations. We consider 8 different optimization formulations for computing a single sparse loading vector:
- ** 
- **
- **
- **
- **
- **
- **
- **

See [paper](https://arxiv.org/abs/1901.09997) for details.
This is a Python software package and C++  software package for SPCA.

This is a Python software package for solving a toy classification problem using neural networks. More specifically, the user can select one of two methods:
- **sampled LBFGS (S-LBFGS)**,
- **sampled LSR1 (S-LSR1)**,

to solve the problem described below. See [paper](https://arxiv.org/pdf/1212.4137.pdf) for details.

Note, the code is extendible to solving other deep learning problems (see comments below).


## Problem


## Citation
If you use 24am for your research, please cite:



## Usage Guide
The algorithms can be run using the syntax: ``` python3 main.py --numOfTrials 10 --sparsity 128 --formulation L2var_L0cons```


### Dependencies
* Numpy
* scipy

### Parameters



### Logs & Printing


The output of the 10 runs is:
```
   Explained variance  Explained variance/ Best explained variance  Seed  Target sparsity level s
0           20.086940                                     0.887711     0                      128
1           20.364809                                     0.899991     1                      128
2           21.821211                                     0.964354     2                      128
3           20.421121                                     0.902479     3                      128
4           22.627801                                     1.000000     4                      128
5           20.421779                                     0.902508     5                      128
6           20.364809                                     0.899991     6                      128
7           18.002156                                     0.795577     7                      128
8           18.928917                                     0.836534     8                      128
9           21.821211                                     0.964354     9                      128
```
 
