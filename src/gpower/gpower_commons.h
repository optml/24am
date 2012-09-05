/*
 * gpower_commons.h
 *
 *  Created on: May 8, 2012
 *      Author: taki
 */

#ifndef GPOWER_COMMONS_H_
#define GPOWER_COMMONS_H_

#include "../utils/various.h"

template<typename F>
void getSignleStartingPoint(F* V, F* Z, optimization_settings* settings,
		const unsigned int n, const unsigned int m, int batchshift,
		unsigned int j) {

	if (settings->isConstrainedProblem()) {
		myseed = j + batchshift;
		F tmp_norm = 0;
		for (unsigned int i = 0; i < settings->constrain; i++) {
			unsigned int idx = (int) (n * (F) rand_r(&myseed) / (RAND_MAX));
			if (idx == n)
				idx--;
			//							for (unsigned int i = 0; i < n; i++) {
			//								unsigned int idx = i;
			F tmp = (F) rand_r(&myseed) / RAND_MAX;
			V[idx] = tmp;
			tmp_norm += tmp * tmp;
		}
		cblas_vector_scale(n, V, 1 / sqrt(tmp_norm));
	} else {
		myseed = j + batchshift;
		F tmp_norm = 0;
		for (unsigned int i = 0; i < n; i++) {
			F tmp = (F) rand_r(&myseed) / RAND_MAX;
			tmp = -1 + 2 * tmp;
			V[i] = tmp;
		}
	}

}

template<typename F>
void initialize_starting_points(F* V, F* Z, optimization_settings* settings,
		optimization_statistics* stat,
		const unsigned int number_of_experiments_per_batch,
		const unsigned int n, const unsigned int m, const int ldB, const F* B,
		int batchshift = 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
		getSignleStartingPoint(&V[j * n], &Z[j * m], settings, n, m,
				batchshift, j);
	}
}

template<typename F>
void perform_one_iteration_for_constrained_pca(F* V, F* Z,
		optimization_settings* settings, optimization_statistics* stat,
		const unsigned int number_of_experiments_per_batch,
		const unsigned int n, const unsigned int m, const int ldB, const F* B,
		F* max_errors, value_coordinate_holder<F>* vals,
		std::vector<F>* buffer, unsigned int it, unsigned int statistical_shift) {

	cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans, CblasNoTrans, m,
			number_of_experiments_per_batch, n, 1, B, ldB, V, n, 0, Z, m); // Multiply z = B*V
	//set Z=sgn(Z)
	if (settings->algorithm == L0_constrained_L1_PCA || settings->algorithm
			== L1_constrained_L1_PCA) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
			vals[j].tmp = cblas_l1_norm(m, &Z[m * j], 1);
			vector_sgn(&Z[m * j], m);//y=sgn(y)
		}
	}
	cblas_matrix_matrix_multiply(CblasColMajor, CblasTrans, CblasNoTrans, n,
			number_of_experiments_per_batch, m, 1, B, ldB, Z, m, 0, V, n);// Multiply V = B'*z

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
		F fval_current = 0;
		if (settings->algorithm == L0_constrained_L2_PCA || settings->algorithm
				== L1_constrained_L2_PCA) {
			fval_current = cblas_l2_norm(m, &Z[m * j], 1);
		}
		F norm_of_x;
		if (settings->isL1ConstrainedProblem()) {
			norm_of_x = soft_tresholding(&V[n * j], n, settings->constrain,
					buffer[j]); // x = S_w(x)
		} else {
			norm_of_x = k_hard_tresholding(&V[n * j], n, settings->constrain,
					buffer[j], settings); // x = T_k(x)
		}

		cblas_vector_scale(n, &V[j * n], 1 / norm_of_x);
		if (settings->algorithm == L0_constrained_L1_PCA || settings->algorithm
				== L1_constrained_L1_PCA) {
			fval_current = vals[j].tmp;
		}
		F tmp_error = computeTheError(fval_current, vals[j].val, settings);
		vals[j].current_error = tmp_error;
		//Log end of iteration for given point
		if (settings->get_it_for_all_points && termination_criteria(tmp_error,
				it, settings) && stat->iters[statistical_shift + j] == -1) {
			stat->iters[statistical_shift + j] = it;
			stat->cardinalities[statistical_shift + j] = vector_get_nnz(
					&V[j * n], n);
		} else if (settings->get_it_for_all_points && !termination_criteria(
				tmp_error, it, settings) && stat->iters[statistical_shift + j]
				!= -1) {
			stat->iters[j + statistical_shift] = -1;
		}
		//---------------
		if (max_errors[my_thread_id] < tmp_error)
			max_errors[my_thread_id] = tmp_error;
		vals[j].val = fval_current;
	}
}

template<typename F>
void perform_one_iteration_for_penalized_pca(F* V, F* Z,
		optimization_settings* settings, optimization_statistics* stat,
		const unsigned int number_of_experiments_per_batch,
		const unsigned int n, const unsigned int m, const int ldB, const F* B,
		F* max_errors, value_coordinate_holder<F>* vals, unsigned int it,
		unsigned int statistical_shift) {
	//scale Z
	cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans, CblasNoTrans, m,
			number_of_experiments_per_batch, n, 1, B, ldB, V, n, 0, Z, m); // Multiply z = B*w
	if (settings->algorithm == L0_penalized_L1_PCA || settings->algorithm
			== L1_penalized_L1_PCA) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
			vector_sgn(&Z[m * j], m);
		}
	} else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
			F tmp_norm = cblas_l2_norm(m, &Z[m * j], 1);
			cblas_vector_scale(m, &Z[j * m], 1 / tmp_norm);
		}
	}
	cblas_matrix_matrix_multiply(CblasColMajor, CblasTrans, CblasNoTrans, n,
			number_of_experiments_per_batch, m, 1, B, ldB, Z, m, 0, V, n);// Multiply v = B'*z
	if (settings->isL1PenalizedProblem()) {
		L1_penalized_tresholding(number_of_experiments_per_batch, n, V,
				settings, max_errors, vals, stat, it);
	} else {
		L0_penalized_tresholding(number_of_experiments_per_batch, n, V,
				settings, max_errors, vals, stat, it);
	}

}

#endif /* GPOWER_COMMONS_H_ */
