# Alternating maximization: Unifying framework for 8 sparse PCA formulations and efficient parallel codes

Authors: [Peter Richtárik](https://richtarik.org/), [Majid Jahani](http://coral.ise.lehigh.edu/maj316/), [Selin Damla Ahipasaoglu](https://esd.sutd.edu.sg/people/faculty/selin-damla-ahipasaoglu) and [Martin Takáč](http://mtakac.com/)


## Introduction
Given a multivariate data set, sparse principal component analysis (SPCA) aims to extract several linear combinations of the variables that together explain the variance in the data as much as possible, while controlling the number of nonzero loadings in these combinations. We consider 8 different optimization formulations for computing a single sparse loading vector:
- L2 variance with L0 constraint  
- L1 variance with L0 constraint
- L2 variance with L1 constraint
- L1 variance with L1 constraint
- L2 variance with L0 penalty
- L1 variance with L0 penalty
- L2 variance with L1 penalty
- L1 variance with L1 penalty

See [paper](https://arxiv.org/pdf/1212.4137.pdf) for details.
This is a Python software package and C++  software package for SPCA.

## Citation
If you use 24am for your research, please cite:
```
@misc{richtrik2012alternating,
    title={Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes},
    author={Peter Richtárik and Majid Jahani and Selin Damla Ahipaşaoğlu and Martin Takáč},
    year={2012},
    eprint={1212.4137},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```
## Usage Guide
The algorithms can be run using the syntax: ``` python3 main.py --numOfTrials 10 --sparsity 128 --formulation L2var_L0cons```


### Dependencies
* Numpy
* scipy

### Parameters
```
 --formulation',     default='L2var_L0cons',
                     help='L2var_L0cons, L1var_L0cons,
                           L2var_L1cons, L1var_L1cons,
                           L2var_L0pen,  L1var_L0pen,
                           L2var_L1pen,  L1var_L1pen'
 --dataname',        default = 'ATandT_Database_of_Faces', type=str,
                     help="AT&T Database of Faces"
 --dataDir',         default = './data/', type=str, help="data directory"
 --resultsDir',      default = './results/', type=str, help="log directory"
 --seed',            default = 1, type=int, help="random seed"
 --density_of_SP',   default = 1, type=float, help="density of starting Point"
 --sparsity',        default = 16, type=int, help="sparsity target"
 --tol',             default = 1e-6, type=float, help="tolerance"
 --maxIter',         default = 200, type=int, help="max num of iterations"
 --numOfTrials',     default = 10, type=int, help="num Of trials"
 --stabilityIter',   default = 30, type=int, help="stability of gamma"
 --incDelta',        default = 1e-3, type=float, help="rate of delta increase"
 ```


### Logs & Printing
All logs are stored in ```.pkl``` file in ```./results/ ```. For example the log file can be as:
```L2var_L0cons_expectedVar_numTraial_10_sparsity_128.pkl```.

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
 
