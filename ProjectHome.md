# 24 parallel sparse PCA codes #

The library supports 8 optimization formulations of Sparse PCA: <br>  <b>L1/L2 variance</b> x <b>L0/L1 sparsity inducing norm</b>  x  norm used as <b>penalty/constraint</b>

It contains parallel codes  for the following architectures<br>
<ul><li>multi-core CPUs (C++, Intel MKL, OpenMP)<br>
</li><li>GPUs (CUDA)<br>
</li><li>computer clusters (C++, BLASC)</li></ul>

Suitable for big data PCA and sparse PCA.<br>
<br>
For more info please see our <a href='Solvers.md'>Wiki</a>.<br>
<br>
<br>
<hr />
Based on the paper:<br>
<br>
Peter Richtarik, Martin Takac and Selin D. Ahipasaoglu, <a href='http://arxiv.org/abs/1212.4137'>Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes</a>, arXiv:1212.4137, December 2012<br>
<hr />



