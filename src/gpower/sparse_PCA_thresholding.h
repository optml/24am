/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M. Takac and S. Damla Ahipasaoglu 
 *    "Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 * 
 */


#ifndef SPARSE_PCA_thresholdING_H_
#define SPARSE_PCA_thresholdING_H_

#include "../utils/openmp_helper.h"
#include "../class/optimization_statistics.h"
#include "../class/optimization_settings.h"
#include "../utils/various.h"


template<typename F>
void L1_penalized_thresholding(const unsigned int number_of_experiments,
		const unsigned int n, F* V, const SolverStructures::OptimizationSettings* optimizationSettings,
		F* max_errors, ValueCoordinateHolder<F>* vals,
		SolverStructures::OptimizationStatistics* optimizationStatistics, const unsigned int it,unsigned int optimizationStatisticsistical_shift=0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int j = 0; j < number_of_experiments; j++) {
		F fval_current = 0;
		for (unsigned i = 0; i < n; i++) {
			F const tmp = V[n * j + i];
			F tmp2 = abs(tmp) - optimizationSettings->penalty;
			if (tmp2 > 0) {
				fval_current += tmp2 * tmp2;
				V[n * j + i] = tmp2 * sgn(tmp);
			} else {
				V[n * j + i] = 0;
			}
		}
		fval_current = sqrt(fval_current);
		F tmp_error = computeTheError(fval_current, vals[j].val, optimizationSettings);
		vals[j].current_error=tmp_error;
		if (max_errors[my_thread_id] < tmp_error)
			max_errors[my_thread_id] = tmp_error;
		vals[j].val = fval_current;
		//Log end of iteration for given point
		if (optimizationSettings->storeIterationsForAllPoints && termination_criteria(tmp_error,
				it, optimizationSettings) && optimizationStatistics->iters[j+optimizationStatisticsistical_shift] == -1) {
			optimizationStatistics->iters[j+optimizationStatisticsistical_shift] = it;
			optimizationStatistics->cardinalities[j+optimizationStatisticsistical_shift] = vector_get_nnz(&V[j * n], n);
		} else if (optimizationSettings->storeIterationsForAllPoints && !termination_criteria(
				tmp_error, it, optimizationSettings) && optimizationStatistics->iters[optimizationStatisticsistical_shift+j] != -1) {
			optimizationStatistics->iters[j+optimizationStatisticsistical_shift] = -1;
		}
		//---------------
	}

}

template<typename F>
void L0_penalized_thresholding(const unsigned int number_of_experiments,
		const unsigned int n, F* V, const SolverStructures::OptimizationSettings* optimizationSettings,
		F* max_errors, ValueCoordinateHolder<F>* vals,
		SolverStructures::OptimizationStatistics* optimizationStatistics, const unsigned int it,unsigned int optimizationStatisticsistical_shift=0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int j = 0; j < number_of_experiments; j++) {
		F fval_current = 0;
		for (unsigned i = 0; i < n; i++) {
			F const tmp = V[n * j + i];
			F tmp2 = (tmp * tmp - optimizationSettings->penalty);
			if (tmp2 > 0) {
				fval_current += tmp2;
			} else {
				V[n * j + i] = 0;
			}
		}
		F tmp_error = computeTheError(fval_current, vals[j].val, optimizationSettings);
		vals[j].current_error=tmp_error;
		if (max_errors[my_thread_id] < tmp_error)
			max_errors[my_thread_id] = tmp_error;
		vals[j].val = fval_current;
		//Log end of iteration for given point
		if (optimizationSettings->storeIterationsForAllPoints && termination_criteria(tmp_error,
				it, optimizationSettings) && optimizationStatistics->iters[j+optimizationStatisticsistical_shift] == -1) {
			optimizationStatistics->cardinalities[optimizationStatisticsistical_shift+j] = vector_get_nnz(&V[j * n], n);
			optimizationStatistics->iters[j+optimizationStatisticsistical_shift] = it;
		} else if (optimizationSettings->storeIterationsForAllPoints && !termination_criteria(
				tmp_error, it, optimizationSettings) && optimizationStatistics->iters[optimizationStatisticsistical_shift+j] != -1) {
			optimizationStatistics->iters[j+optimizationStatisticsistical_shift] = -1;
		}
		//---------------
	}
}


#endif /* SPARSE_PCA_thresholdING_H_ */
