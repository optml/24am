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

#ifndef GPOWER_COMMONS_H_
#define GPOWER_COMMONS_H_

#include "../utils/various.h"

// this function generate initial points
template<typename F>
void getSignleStartingPoint(F* V, F* Z,
		SolverStructures::OptimizationSettings* optimizationSettings,
		const unsigned int n, const unsigned int m, int batchshift,
		unsigned int j) {
	if (optimizationSettings->isConstrainedProblem()) {
		myseed = j + batchshift;
		F tmp_norm = 0;
		for (unsigned int i = 0; i < optimizationSettings->constraintParameter; i++) {
			unsigned int idx = (int) (n * (F) rand_r(&myseed) / (RAND_MAX));
			if (idx == n)
				idx--;
			F tmp = (F) rand_r(&myseed) / RAND_MAX;
			V[idx] = tmp;
			tmp_norm += tmp * tmp;
		}
		cblas_vector_scale(n, V, 1 / sqrt(tmp_norm));
	} else {
		myseed = j + batchshift;
		for (unsigned int i = 0; i < n; i++) {
			F tmp = (F) rand_r(&myseed) / RAND_MAX;
			tmp = -1 + 2 * tmp;
			V[i] = tmp;
		}
	}

}

//initialize starting points
template<typename F>
void initialize_totalStartingPoints(F* V, F* Z,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics,
		const unsigned int number_of_experiments_per_batch,
		const unsigned int n, const unsigned int m, const int ldB, const F* B,
		int batchshift = 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
		getSignleStartingPoint(&V[j * n], &Z[j * m], optimizationSettings, n, m, batchshift,
				j);
	}
}

// do one iteration for constrained PCA
template<typename F>
void perform_one_iteration_for_constrained_pca(F* V, F* Z,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics,
		const unsigned int number_of_experiments_per_batch,
		const unsigned int n, const unsigned int m, const int ldB, const F* B,
		F* max_errors, ValueCoordinateHolder<F>* vals, std::vector<F>* buffer,
		unsigned int it, unsigned int optimizationStatisticsistical_shift) {
	cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans, CblasNoTrans, m,
			number_of_experiments_per_batch, n, 1, B, ldB, V, n, 0, Z, m); // Multiply z = B*V
	//set Z=sgn(Z)
	if (optimizationSettings->formulation == SolverStructures::L0_constrained_L1_PCA
			|| optimizationSettings->formulation
					== SolverStructures::L1_constrained_L1_PCA) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
			vals[j].tmp = cblas_l1_norm(m, &Z[m * j], 1);
			vector_sgn(&Z[m * j], m);	//y=sgn(y)
		}
	}
	cblas_matrix_matrix_multiply(CblasColMajor, CblasTrans, CblasNoTrans, n,
			number_of_experiments_per_batch, m, 1, B, ldB, Z, m, 0, V, n);// Multiply V = B'*z

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
		F fval_current = 0;
		if (optimizationSettings->formulation == SolverStructures::L0_constrained_L2_PCA
				|| optimizationSettings->formulation
						== SolverStructures::L1_constrained_L2_PCA) {
			fval_current = cblas_l2_norm(m, &Z[m * j], 1);
		}
		F norm_of_x;
		if (optimizationSettings->isL1ConstrainedProblem()) {
			norm_of_x = soft_thresholding(&V[n * j], n, optimizationSettings->constraintParameter,
					buffer[j], optimizationSettings); // x = S_w(x)
		} else {
			norm_of_x = k_hard_thresholding(&V[n * j], n, optimizationSettings->constraintParameter,
					buffer[j], optimizationSettings); // x = T_k(x)
		}

		cblas_vector_scale(n, &V[j * n], 1 / norm_of_x);
		if (optimizationSettings->formulation == SolverStructures::L0_constrained_L1_PCA
				|| optimizationSettings->formulation
						== SolverStructures::L1_constrained_L1_PCA) {
			fval_current = vals[j].tmp;
		}
		F tmp_error = computeTheError(fval_current, vals[j].val, optimizationSettings);
		vals[j].current_error = tmp_error;
		//Log end of iteration for given point
		if (optimizationSettings->storeIterationsForAllPoints
				&& termination_criteria(tmp_error, it, optimizationSettings)
				&& optimizationStatistics->iters[optimizationStatisticsistical_shift + j] == -1) {
			optimizationStatistics->iters[optimizationStatisticsistical_shift + j] = it;
			optimizationStatistics->cardinalities[optimizationStatisticsistical_shift + j] = vector_get_nnz(
					&V[j * n], n);
		} else if (optimizationSettings->storeIterationsForAllPoints
				&& !termination_criteria(tmp_error, it, optimizationSettings)
				&& optimizationStatistics->iters[optimizationStatisticsistical_shift + j] != -1) {
			optimizationStatistics->iters[j + optimizationStatisticsistical_shift] = -1;
		}
		//---------------
		if (max_errors[my_thread_id] < tmp_error)
			max_errors[my_thread_id] = tmp_error;
		vals[j].val = fval_current;
	}
}

// do one iteration for penalize PCA
template<typename F>
void perform_one_iteration_for_penalized_pca(F* V, F* Z,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics,
		const unsigned int number_of_experiments_per_batch,
		const unsigned int n, const unsigned int m, const int ldB, const F* B,
		F* max_errors, ValueCoordinateHolder<F>* vals, unsigned int it,
		unsigned int optimizationStatisticsistical_shift) {
	//scale Z
	cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans, CblasNoTrans, m,
			number_of_experiments_per_batch, n, 1, B, ldB, V, n, 0, Z, m); // Multiply z = B*w
	if (optimizationSettings->formulation == SolverStructures::L0_penalized_L1_PCA
			|| optimizationSettings->formulation == SolverStructures::L1_penalized_L1_PCA) {
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
			number_of_experiments_per_batch, m, 1, B, ldB, Z, m, 0, V, n); // Multiply v = B'*z
	if (optimizationSettings->isL1PenalizedProblem()) {
		L1_penalized_thresholding(number_of_experiments_per_batch, n, V,
				optimizationSettings, max_errors, vals, optimizationStatistics, it);
	} else {
		L0_penalized_thresholding(number_of_experiments_per_batch, n, V,
				optimizationSettings, max_errors, vals, optimizationStatistics, it);
	}

}

#endif /* GPOWER_COMMONS_H_ */
