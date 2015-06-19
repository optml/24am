# Introduction #

## Three Solvers ##

The codes are packaged as 3 solvers depending on the intended computing architecture:
  1. single/multicore workstation,
  1. GPU,
  1. cluster.

They can be run in console mode or from within C++.

The **Multicore Solver** is suitable for desktop applications wishing to utilize multicore capability of current processors. It is appropriate for small or medium sized problems (depending on RAM). Speedup is proportional to the number of processors.

The **GPU Solver** is most appropriate for problems with data up several GB in size (GPUs have limited memory) where speed or the amount of expected variance is of crucial importance. This is due to the fact that modern GPUs have thousands of cores - the speedup of our GPU solver, compared to single-core C++ solver, can be several orders of magnitude.

The **Cluster Solver** is most appropriate for big data applications, where the data is too large to be stored on a single computer. Our Cluster solver works with data distributed among the nodes of a cluster, and utilizes the parallel processing power of each node. Problems with data TB matrices can be solved efficiently.

## Eight Formulations ##

Each solver comes with several options, most notably the optimization formulation one wishes to use. There are 8 optimization formulations for finding a single sparse dominant principal component (PC) available for each of the three solvers; this results in 24 codes. The formulations arise from the combination of the following factors: one can measure variance using two norms (standard L2 and robust L1) and use two sparsity inducing norms (L0 and L1) in two different ways (as a constraint/penalty).

The penalized formulations are uniquely characterized by the choice of the two norms (for measuring variance and inducing sparsity) and a nonnegative penalty parameter. This parameter is denoted 'gamma' in (1), 'g' in console mode and 'penaltyParameter' in C++. Increasing the penalty parameter will lead to a sparser principal loading vector.

The constrained formulations are uniquely characterized by the choice of the two norms (for measuring variance and inducing sparsity) and a positive integer parameter. This parameter is denoted 's' in (1) and in the console mode and 'contraintParameter' in C++. Decreasing the constraint parameter will lead to a sparser principal loading vector. In the case of the two L0 constrained formulations, 's' is a hard bound on the number of nonzero loadings.

Please refer to Section 1 of (1) for details.

## Parallelization ##

Each solver runs several AM subroutines in parallel, started from different starting points, with the aim to obtain sparse PC explaining more variance. The number of subroutines is controlled by the option 'l' in the console mode and variable 'totalStartingPoints' in C++.

Especially in cases where high sparsity is sought, the case l=1 typically give a sparse principal component explaining less variance than possible (see Figure 2 in (1)). Therefore, it may be preferable to run the AM subroutine from several random starting points and accept the best sparse PC, that is, the one explaining most variance.

The solvers generate 'l' random starting points and group them into batches of size 'r' (both l and r are solver options), executing each batch simultaneously, in parallel.

Parallelization speedup is obtained at two levels: by streamlining the linear algebra involved inside every single AM thread, and also the linear algebra arising from running several AM threads simultaneously.

Please refer to Section 4 of (1) for details.



## Solver Options ##

The following table describes which options can be used for which solver (X means that the option is not implemented, NA means that the option is not applicable).

<table border='1'>
<tr>
<blockquote><th>Description</th>
<th>Console option</th>
<th>C++ variable</th></blockquote>

<blockquote><th>Default value</th>
<th>Multicore</th>
<th>GPU</th>
<th>Cluster</th>
</tr></blockquote>

<tr>
<blockquote><td><b>Input file.</b> Path to file containing the data matrix.</td>
<td> -i </td>
<td><code>char* inputFile</code></td></blockquote>

<blockquote><td> required </td>
<td> ok </td><td> ok </td><td> ok </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Output file.</b> Path to file where output will be stored.</td>
<td>-o </td>
<td><code>char* outputFilePath</code></td></blockquote>

<blockquote><td>required</td>
<td> ok </td><td> ok </td><td> ok </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Formulation.</b> Specifies which of the 8 SPCA optimization formulations will be solved (see also Table 1 in (1)):<br>
</blockquote><ol><li>L0 constrained L2 variance SPCA<br>
</li><li>L0 constrained L1 variance SPCA<br>
</li><li>L1 constrained L2 variance SPCA<br>
</li><li>L1 constrained L1 variance SPCA<br>
</li><li>L0 penalized L2 variance SPCA<br>
</li><li>L0 penalized L1 variance SPCA<br>
</li><li>L1 penalized L2 variance SPCA<br>
</li><li>L1 penalized L1 variance SPCA</td>
</li></ol><blockquote><td>-f</td>
<td><code>enum SPCA_Formulation formulation</code></td></blockquote>

<blockquote><td>required</td>
<td> ok </td><td> ok </td><td> ok </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Penalty parameter.</b> Relevant only in formulations involving a penalty.</td>
<td>-g</td>
<td><code>double penaltyParameter</code></td></blockquote>

<blockquote><td>0</td>
<td> ok </td><td> ok </td><td> ok </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Constraint parameter.</b> Relevant only in formulations involving a constraint.</td>
<td>-s</td>
<td><code>unsigned int constraintParameter</code></td></blockquote>

<blockquote><td>10</td>
<td> ok </td><td> ok </td><td> ok </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Parallelization.</b> Number of AM subroutines run by the solver. Each subroutine is run from a random starting point. More subroutines results in more explained variance, especially if a highly sparse PC is sought.</td>
<td>-l</td>
<td><code>int totalStartingPoints</code></td>
<td>64</td>
<td> ok </td><td> ok </td><td> ok </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Batch size.</b> The number of AM subroutines solved in parallel at any given point in time. Ideally, 'l' is an integer multiple of 'r'. </td>
<td>-r</td>
<td><code>int batchSize</code></td></blockquote>

<blockquote><td>64</td>
<td> ok </td><td>   </td><td> X </td>
</tr></blockquote>

<tr>
<blockquote><td><b>On-the-fly (OTF).</b> If set to true, a smart dynamic replacement strategy is employed which accelerates the solver. Refer to Section 4.3 of (1) for more detail. </td>
<td>-u</td>
<td><code>bool useOTF</code></td></blockquote>

<blockquote><td>false</td>
<td> ok </td><td>   </td><td> X </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Stopping criterion 1.</b> Maximum number of iterations for each AM subroutine.</td>
<td>-m</td>
<td><code>int maximumIterations</code></td></blockquote>

<blockquote><td>20</td>
<td> ok </td><td> ok </td><td> ok </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Stopping criterion 2.</b> AM subroutine stops if the relative increase of the objective value during an iteration is smaller than 'tolerance'.</td>
<td>-t</td>
<td><code>double tolerance</code></td></blockquote>

<blockquote><td>0.01</td>
<td> ok </td><td> ok </td><td> ok </td>
</tr></blockquote>


<tr>
<blockquote><td>Specifies the x-dimension of virtual 2D computational grid for BLASC </td>
<td>-x</td>
<td><code>int distributedRowGridFile</code></td></blockquote>

<blockquote><td></td>
<td> NA </td><td> NA </td><td> ok </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Verbose.</b> If true, solver will print messages during execution. </td>
<td>-v</td>
<td><code>bool  verbose</code></td></blockquote>

<blockquote><td>false</td>
<td> ok </td><td> ok </td><td> ok </td>
</tr></blockquote>

<tr>
<blockquote><td><b>Double precision.</b> Specifies whether double or single precision will be used in computations. </td>
<td>-d</td>
<td><code>bool useDoublePrecision</code></td></blockquote>

<blockquote><td>false</td>
<td> ok </td><td> ok </td><td>  </td>
</tr></blockquote>

</table>



