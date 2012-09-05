/*
 * sparse_PCA_tresholding.h
 *
 *  Created on: Apr 12, 2012
 *      Author: taki
 */

#ifndef SPARSE_PCA_TRESHOLDING_H_
#define SPARSE_PCA_TRESHOLDING_H_

#include "../utils/openmp_helper.h"
#include "../class/optimization_statistics.h"
#include "../class/optimization_settings.h"
#include "../utils/various.h"


template<typename F>
void L1_penalized_tresholding(const unsigned int number_of_experiments,
		const unsigned int n, F* V, const optimization_settings* settings,
		F* max_errors, value_coordinate_holder<F>* vals,
		optimization_statistics* stat, const unsigned int it,unsigned int statistical_shift=0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int j = 0; j < number_of_experiments; j++) {
		F fval_current = 0;
		for (unsigned i = 0; i < n; i++) {
			F const tmp = V[n * j + i];
			F tmp2 = abs(tmp) - settings->penalty;
			if (tmp2 > 0) {
				fval_current += tmp2 * tmp2;
				V[n * j + i] = tmp2 * sgn(tmp);
			} else {
				V[n * j + i] = 0;
			}
		}
		fval_current = sqrt(fval_current);
		F tmp_error = computeTheError(fval_current, vals[j].val, settings);
		vals[j].current_error=tmp_error;
		if (max_errors[my_thread_id] < tmp_error)
			max_errors[my_thread_id] = tmp_error;
		vals[j].val = fval_current;
		//Log end of iteration for given point
		if (settings->get_it_for_all_points && termination_criteria(tmp_error,
				it, settings) && stat->iters[j+statistical_shift] == -1) {
			stat->iters[j+statistical_shift] = it;
			stat->cardinalities[j+statistical_shift] = vector_get_nnz(&V[j * n], n);
		} else if (settings->get_it_for_all_points && !termination_criteria(
				tmp_error, it, settings) && stat->iters[statistical_shift+j] != -1) {
			stat->iters[j+statistical_shift] = -1;
		}
		//---------------
	}

}

template<typename F>
void L0_penalized_tresholding(const unsigned int number_of_experiments,
		const unsigned int n, F* V, const optimization_settings* settings,
		F* max_errors, value_coordinate_holder<F>* vals,
		optimization_statistics* stat, const unsigned int it,unsigned int statistical_shift=0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int j = 0; j < number_of_experiments; j++) {
		F fval_current = 0;
		for (unsigned i = 0; i < n; i++) {
			F const tmp = V[n * j + i];
			F tmp2 = (tmp * tmp - settings->penalty);
			if (tmp2 > 0) {
				fval_current += tmp2;
			} else {
				V[n * j + i] = 0;
			}
		}
		F tmp_error = computeTheError(fval_current, vals[j].val, settings);
		vals[j].current_error=tmp_error;
		if (max_errors[my_thread_id] < tmp_error)
			max_errors[my_thread_id] = tmp_error;
		vals[j].val = fval_current;
		//Log end of iteration for given point
		if (settings->get_it_for_all_points && termination_criteria(tmp_error,
				it, settings) && stat->iters[j+statistical_shift] == -1) {
			stat->cardinalities[statistical_shift+j] = vector_get_nnz(&V[j * n], n);
			stat->iters[j+statistical_shift] = it;
		} else if (settings->get_it_for_all_points && !termination_criteria(
				tmp_error, it, settings) && stat->iters[statistical_shift+j] != -1) {
			stat->iters[j+statistical_shift] = -1;
		}
		//---------------
	}
}


#endif /* SPARSE_PCA_TRESHOLDING_H_ */