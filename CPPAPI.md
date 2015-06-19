## Single-core / Multi-core ##



```
template<typename F>
F SPCASolver::MulticoreSolver::denseDataSolver(
                  const F * B, 
                  const int ldB, 
                  F * x, 
                  const unsigned int m,
		  const unsigned int n, 
                  SolverStructures::OptimizationSettings* optimizationSettings,
		  SolverStructures::OptimizationStatistics* optimizationStatistics)
```



## CUDA GPU ##

```
template<typename F>
int SPCASolver::GPUSolver::denseDataSolver(
                cublasHandle_t &handle, 
                const unsigned int m,
		const unsigned int n, 
                thrust::device_vector<F> &d_B,
		thrust::host_vector<F>& h_x,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics,
		const unsigned int LD_M, 
                const unsigned int LD_N)
```

## Cluster ##
```
template<typename F>
void SPCASolver::DistributedSolver::denseDataSolver(
		SPCASolver::DistributedClasses::OptimizationData<F>& optimizationDataInstance,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics)
```