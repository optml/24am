In this document we explain how to build solvers for the following 3 architectures: Multicore, GPU and Cluster




# Guide: compiling console solvers #

The easiest way to use our library is to compile and run a solver interface which can be run from console.


## Single-core / Multi-core ##

**Prerequisites**

In order to compile the code you need the following libraries
  * [GSL](http://www.gnu.org/software/gsl/) - we tested the code with GSL v1.9 and v1.15. The main usage of this library is for **cblas** which is used for linear algebra (mainly matrix-matrix multiplication).

Optionally, if you wish to fully utilize multi-core architectures, you can install parallel implementation of cblas interface. You can choose, for instance,
  * [Intel MKL](http://software.intel.com/en-us/intel-mkl) which is not open source but has non-commercial license
  * [GoToBlas2](http://www.tacc.utexas.edu/tacc-projects/gotoblas2) which is free to use (BSD license).

**Configuration of build script**

We assume that you downloaded the code and "/" is  root of the library.
In **/Makefile** set the paths to GSL. For example, if your GSL is installed in
_/exports/applications/apps/gsl/1.9_ it should looks like
```
# Path to GSL library
GSL_INCLUDE = -I/exports/applications/apps/gsl/1.9/include
GSL_LIB= -L/exports/applications/apps/gsl/1.9/lib
```

For Intel MKL multi-core support you have to specify MKL location, e.g.,
```
# Path to Intel MKL + Linking settings
MKLROOT = /exports/applications/apps/SL5/intel/MKL/10.2.3.029
MKL_MULTICORE_LIB =    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lgsl -lm  
```
The  **MKL\_MULTICORE\_LIB** dependens on architecture and compiler used. We highly recommend the use of [Intel® Math Kernel Library Link Line Advisor](http://software.intel.com/sites/products/mkl/) to find out how to link against Intel MKL.
Finally, enable Intel MKL in linking process
```
#Choose CBLAS interface GSL/MKL
#LIBS_BLAS= $(LIBS_GSL)
#BLAS_LIB= $(GSL_LIB)
LIBS_BLAS= $(LIBS_MKL)
BLAS_LIB= $(MKL_MULTICORE_LIB)

```


**Building a binary**

Now we are ready to build the binary. Just type in "/" following
```
make multicore_console
```
and the binary is build into `build/multicore_console`.
To test the binary you can call from "/" e.g.
```
./build/multicore_console -i datasets/small.csv  -o results/small.txt -v true -d double -f 1 -s 3
```
See [ConsoleParameters](ConsoleParameters.md) for more information about input parameters.

## CUDA GPU ##

**Prerequisites** The GPU version requires that you have CUDA GPU card (we tested CUDA 4.0+, versions 4.2, 4.0rc2, 4.0). You need the {nvcc} compiler. We tested on the Tesla M2050 card.

**Configuration of build script**

Set **CUDA\_INSTALL\_PATH** in `/gpu.mk`. You will find more guides in that file. As an example in our case we have
```
CUDA_INSTALL_PATH= /exports/applications/apps/cuda/rhel5/4.2/cuda
```

**Building a binary**

Now we are ready to build the binary. Just type in "/" following
```
make gpu_console
```
and the binary is built into `build/gpu_console`.
To run a uni-test, which solves a collection of problems using the GPU and Multi-core solvers, type
```
make gpu_unit_test
```
and you should obtain output as follows
```
Double test
CUBLAS initialized.
Test 0 L0_penalized_L1_PCA 299.61  299.61
Test 1 L0_penalized_L2_PCA 0.374924  0.374924
Test 2 L1_penalized_L1_PCA 17.3093  17.3093
Test 3 L1_penalized_L2_PCA 0.61231  0.61231
Test 4 L0_constrained_L1_PCA 4.65016  4.65016
Test 5 L0_constrained_L2_PCA 0.165733  0.165733
Test 6 L1_constrained_L1_PCA 4.66424  4.66424
Test 7 L1_constrained_L2_PCA 0.166199  0.166199
```


## Cluster ##

**Prerequisites**

Cluster version uses PBLAS and requires **Intel® Math Kernel Library**. We tested our code with v11.0. Note that Intel MKL is not open-source but there is a [non-commercial license available](http://software.intel.com/en-us/non-commercial-software-development)

**Configuration of build script**

Set **MKL ROOT** in `/Makefile`
and **MKL\_LIBS** in `cluster.mk`. See [multi-core setup section](Building#Single-core_/_Multi-core.md).

**Building a binary**

Now we are ready to build the binary. Just type in "/" following
```
make cluster_console
```
and the binary is built into `build/cluster_console`.
To perform a uni-test, which solves a collection of problems using the Cluster and Multi-core solvers,  type
```
make cluster_unit_test
```
and you should obtain output as follows
```
Test 0 L0_penalized_L1_PCA 299.61  299.61
Test 1 L0_penalized_L2_PCA 0.374924  0.374924
Test 2 L1_penalized_L1_PCA 17.3093  17.3093
Test 3 L1_penalized_L2_PCA 0.61231  0.61231
Test 4 L0_constrained_L1_PCA 4.65016  4.65016
Test 5 L0_constrained_L2_PCA 0.165722  0.165706
Test 6 L1_constrained_L1_PCA 4.66424  4.66424
Test 7 L1_constrained_L2_PCA 0.166199  0.166199
```