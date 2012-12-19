/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M. Takac and S. Damla Ahipasaoglu 
 *    "Alternating Maximization: Unified Framework and 24 Parallel Codes for L1 and L2 based Sparse PCA"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 * 
 */

#ifndef SPARSE_PCA_SOLVER_CSC_H_
#define SPARSE_PCA_SOLVER_CSC_H_
#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
#include "../utils/my_cblas_wrapper.h"
#include "my_sparse_cblas_wrapper.h"
#include "../utils/thresh_functions.h"
#include "../utils/timer.h"

#include "sparse_PCA_thresholding.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

template<typename F>
void printDescriptions(F* x, int length, const char* description,
		SolverStructures::OptimizationStatisticsistics* optimizationStatistics, ofstream& stream) {
	FILE * fin = fopen(description, "r");
	char buffer[1000];
	if (fin == NULL) {
		printf("File not found\n");
		exit(1);
	} else {
		printf("------\n");
		cout << "Objective value: " << optimizationStatistics->fval << endl;
		stream << "Objective value: " << optimizationStatistics->fval << endl;
		for (int k = 0; k < length; k++) {
			fscanf(fin, "%s\n", &buffer);
			if (x[k] != 0) {
				stream << k << ":" << buffer << endl;
				printf("%d: %s\n", k, buffer);
			}

		}
		fclose(fin);
	}
}

namespace SPCASolver {

/*
 * Matrix B is stored in column order (Fortran Based)
 */
template<typename F>
F sparse_PCA_solver_CSC(F * B_CSC_Vals, int* B_CSC_Row_Id, int* B_CSC_Col_Ptr,
		F * x, int m, int n, SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatisticsistics* optimizationStatistics, bool doMean,
		F * means) {
	int number_of_experiments = optimizationSettings->starting_points;
	ValueCoordinateHolder<F>* vals = (ValueCoordinateHolder<F>*) calloc(
			number_of_experiments, sizeof(ValueCoordinateHolder<F> ));
	F * Z = (F*) calloc(m * number_of_experiments, sizeof(F));
	F * V = (F*) calloc(n * number_of_experiments, sizeof(F));
	F * ZZ = (F*) calloc(m * number_of_experiments, sizeof(F));
	F * VV = (F*) calloc(n * number_of_experiments, sizeof(F));

	optimizationStatistics->it = optimizationSettings->max_it;
	// Allocate vector for optimizationStatistics to return which point needs how much iterations
	if (optimizationSettings->storeIterationsForAllPoints) {
		optimizationStatistics->iters.resize(optimizationSettings->starting_points, -1);
	}
	F FLOATING_ZERO = 0;
	if (optimizationSettings->isConstrainedProblem()) {
		//				cblas_dscal(n * number_of_experiments, 0, V, 1);
#ifdef _OPENMP
//#pragma omp parallel for
#endif
		for (unsigned int j = 0; j < number_of_experiments; j++) {
			myseed = rand();
			F tmp_norm = 0;
			//			for (unsigned int i = 0; i < n;i++){//optimizationSettings->constrain; i++) {
			//				unsigned int idx = i;

			for (unsigned int i = 0; i < n; i++) {
				unsigned int idx = i;//(int) (n * (F) rand_r(&myseed) / (RAND_MAX));
				if (idx == n)
					idx--;
				//printf("%d\n",idx);

				F tmp = (F) rand_r(&myseed) / RAND_MAX;
				V[j * n + idx] = tmp;
				tmp_norm += tmp * tmp;
			}
			cblas_vector_scale(n, &V[j * n], 1 / sqrt(tmp_norm));
		}

	} else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (unsigned int j = 0; j < number_of_experiments; j++) {
			myseed = j;
			F tmp_norm = 0;
			for (unsigned int i = 0; i < m; i++) {
				F tmp = (F) rand_r(&myseed) / RAND_MAX;
				tmp = -1 + 2 * tmp;
				Z[j * m + i] = tmp;
			}
		}
	}

	F error = 0;
	F max_errors[TOTAL_THREADS];

	F floating_zero = 0;
	F floating_one = 1;
	MKL_INT ONE_MKL_INT = 1;
	char matdescra[6];
	matdescra[0] = 'g';
	matdescra[1] = 'X';
	matdescra[2] = 'X';
	matdescra[3] = 'C';

	std::vector<F>* buffer = (std::vector<F>*) calloc(number_of_experiments,
			sizeof(std::vector<F>));
	if (optimizationSettings->isConstrainedProblem()) {
		for (unsigned int j = 0; j < number_of_experiments; j++) {
			buffer[j].resize(n);
		}
	}

	double start_time_of_iterations = gettime();
	for (unsigned int it = 0; it < optimizationSettings->max_it; it++) {
		for (unsigned int tmp = 0; tmp < TOTAL_THREADS; tmp++) {
			max_errors[tmp] = 0;
		}
		if (optimizationSettings->isConstrainedProblem()) {

			if (doMean) {
				my_mm_multiply(false, m, n, number_of_experiments, B_CSC_Vals,
						B_CSC_Row_Id, B_CSC_Col_Ptr, means, V, Z);
			} else {
				for (int ex = 0; ex < number_of_experiments; ex++) {
					for (int i = 0; i < n; i++)
						VV[i * number_of_experiments + ex] = V[i + ex * n];
				}
				sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_NOTRANS, m,
						number_of_experiments, n, &floating_one, matdescra,
						B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr,
						&B_CSC_Col_Ptr[1], VV, number_of_experiments,
						&floating_zero, ZZ, number_of_experiments);
				for (int ex = 0; ex < number_of_experiments; ex++) {
					for (int i = 0; i < m; i++)
						Z[i + m * ex] = ZZ[number_of_experiments * i + ex];
//						Z[i + ex * m] = ZZ[i * number_of_experiments + ex];
				}

			}
			//set Z=sgn(Z)
			if (optimizationSettings->algorithm == SolverStructures::L0_constrained_L1_PCA
					|| optimizationSettings->algorithm
							== SolverStructures::L1_constrained_L1_PCA) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (unsigned int j = 0; j < number_of_experiments; j++) {
					vals[j].tmp = cblas_l1_norm(m, &Z[m * j], 1);
					vector_sgn(&Z[m * j], m);			//y=sgn(y)
				}
			}

			if (doMean) {
				my_mm_multiply(true, m, n, number_of_experiments, B_CSC_Vals,
						B_CSC_Row_Id, B_CSC_Col_Ptr, means, Z, V);
			} else {
				for (int ex = 0; ex < number_of_experiments; ex++) {
					for (int i = 0; i < m; i++)
						ZZ[number_of_experiments * i + ex] = Z[i + m * ex];
				}
				sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_TRANS, m,
						number_of_experiments, n, &floating_one, matdescra,
						B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr,
						&B_CSC_Col_Ptr[1], ZZ, number_of_experiments,
						&floating_zero, VV, number_of_experiments);
				for (int ex = 0; ex < number_of_experiments; ex++) {
					for (int i = 0; i < n; i++)
						V[i + ex * n] = VV[i * number_of_experiments + ex];
				}

			}
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (unsigned int j = 0; j < number_of_experiments; j++) {
				F fval_current = 0;
				if (optimizationSettings->algorithm
						== SolverStructures::L0_constrained_L2_PCA
						|| optimizationSettings->algorithm
								== SolverStructures::L1_constrained_L2_PCA) {
					fval_current = cblas_l2_norm(m, &Z[m * j], 1);
				}
				F norm_of_x;
				if (optimizationSettings->isL1ConstrainedProblem()) {
					norm_of_x = soft_thresholding(&V[n * j], n,
							optimizationSettings->constrain, buffer[j], optimizationSettings); // x = S_w(x)
				} else {
					norm_of_x = k_hard_thresholding(&V[n * j], n,
							optimizationSettings->constrain, buffer[j], optimizationSettings); // x = T_k(x)
				}

				cblas_vector_scale(n, &V[j * n], 1 / norm_of_x);
				if (optimizationSettings->algorithm
						== SolverStructures::L0_constrained_L1_PCA
						|| optimizationSettings->algorithm
								== SolverStructures::L1_constrained_L1_PCA) {
					fval_current = vals[j].tmp;
				}
				F tmp_error = computeTheError(fval_current, vals[j].val,
						optimizationSettings);
				//Log end of iteration for given point
				if (optimizationSettings->storeIterationsForAllPoints
						&& termination_criteria(tmp_error, it, optimizationSettings)
						&& optimizationStatistics->iters[j] == -1) {
					optimizationStatistics->iters[j] = it;
				} else if (optimizationSettings->storeIterationsForAllPoints
						&& !termination_criteria(tmp_error, it, optimizationSettings)
						&& optimizationStatistics->iters[j] != -1) {
					optimizationStatistics->iters[j] = -1;
				}
				//---------------
				if (max_errors[my_thread_id] < tmp_error)
					max_errors[my_thread_id] = tmp_error;
				vals[j].val = fval_current;
			}
		} else {
			//scale Z
			if (optimizationSettings->algorithm == SolverStructures::L0_penalized_L1_PCA
					|| optimizationSettings->algorithm
							== SolverStructures::L1_penalized_L1_PCA) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (unsigned int j = 0; j < number_of_experiments; j++) {
					vector_sgn(&Z[m * j], m);
				}
			} else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (unsigned int j = 0; j < number_of_experiments; j++) {
					F tmp_norm = cblas_l2_norm(m, &Z[m * j], 1);
					cblas_vector_scale(m, &Z[j * m], 1 / tmp_norm);
				}
			}
			sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_TRANS, m,
					number_of_experiments, n, &floating_one, matdescra,
					B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr, &B_CSC_Col_Ptr[1],
					Z, ONE_MKL_INT, &floating_zero, V, ONE_MKL_INT);

			if (optimizationSettings->isL1PenalizedProblem()) {
				L1_penalized_thresholding(number_of_experiments, n, V, optimizationSettings,
						max_errors, vals, optimizationStatistics, it);
			} else {
				L0_penalized_thresholding(number_of_experiments, n, V, optimizationSettings,
						max_errors, vals, optimizationStatistics, it);
			}

			sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_NOTRANS, m,
					number_of_experiments, n, &floating_one, matdescra,
					B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr, &B_CSC_Col_Ptr[1],
					V, ONE_MKL_INT, &floating_zero, Z, ONE_MKL_INT);
		}
		error =
				max_errors[cblas_vector_max_index(TOTAL_THREADS, max_errors, 1)];
		if (termination_criteria(error, it, optimizationSettings)) {
			optimizationStatistics->it = it;
			break;
		}

	}
	double end_time_of_iterations = gettime();
	//compute corresponding x
	optimizationStatistics->values.resize(optimizationSettings->starting_points);
	int selected_idx = 0;
	F best_value = vals[selected_idx].val;
	optimizationStatistics->values[0] = best_value;
	optimizationStatistics->true_computation_time = (end_time_of_iterations
			- start_time_of_iterations);
	for (unsigned int i = 1; i < number_of_experiments; i++) {
		optimizationStatistics->values[i] = vals[i].val;
		if (vals[i].val > best_value) {
			best_value = vals[i].val;
			selected_idx = i;
		}
	}
	cblas_vector_copy(n, &V[n * selected_idx], 1, x, 1);
	F norm_of_x = cblas_l2_norm(n, x, 1);
	cblas_vector_scale(n, x, 1 / norm_of_x); //Final x
	free(Z);
	free(V);
	free(VV);
	free(ZZ);
	free(vals);
	optimizationStatistics->fval = best_value;
	return best_value;
}

}

#endif /* SPARSE_PCA_SOLVER_H__ */
