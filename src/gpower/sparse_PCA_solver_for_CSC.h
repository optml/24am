/*
//HEADER INFO
 */


#ifndef SPARSE_PCA_SOLVER_CSC_H_
#define SPARSE_PCA_SOLVER_CSC_H_
#include "optimization_settings.h"
#include "optimization_statistics.h"
#include "my_cblas_wrapper.h"
#include "my_sparse_cblas_wrapper.h"
#include "helpers.h"
#include "tresh_functions.h"
#include "sparse_PCA_tresholding.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

template<typename F>
void printDescriptions(F* x, int length, const char* description) {
	FILE * fin = fopen(description, "r");
	char buffer[1000];
	if (fin == NULL) {
		printf("File not found\n");
		exit(1);
	} else {
		printf("------\n");
		for (int k = 0; k < length; k++) {
			fscanf(fin, "%s\n", &buffer);
			if (x[k] != 0) {
				printf("%d: %s\n", k, buffer);
			}

		}
		fclose(fin);
	}
}

/*
 * Matrix B is stored in column order (Fortran Based)
 */
template<typename F>
F sparse_PCA_solver_CSC(F * B_CSC_Vals, int* B_CSC_Row_Id, int* B_CSC_Col_Ptr,
		F * x, int m, int n, optimization_settings* settings,
		optimization_statistics* stat, bool doMean, F * means) {
	int number_of_experiments = settings->starting_points;
	F * Z = (F*) calloc(m * number_of_experiments, sizeof(F));
	value_coordinate_holder<F>* vals = (value_coordinate_holder<F>*) calloc(
			number_of_experiments, sizeof(value_coordinate_holder<F> ));
	F * V = (F*) calloc(n * number_of_experiments, sizeof(F));
	stat->it = settings-> max_it;
	// Allocate vector for stat to return which point needs how much iterations
	if (settings->get_it_for_all_points) {
		stat->iters.resize(settings->starting_points, -1);
	}
	F FLOATING_ZERO = 0;
	if (settings->isConstrainedProblem()) {
		//				cblas_dscal(n * number_of_experiments, 0, V, 1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (unsigned int j = 0; j < number_of_experiments; j++) {
			myseed = rand();
			F tmp_norm = 0;
			//			for (unsigned int i = 0; i < n;i++){//settings->constrain; i++) {
			//				unsigned int idx = i;

			for (unsigned int i = 0; i < settings->constrain; i++) {
				unsigned int idx = (int) (n * (F) rand_r(&myseed) / (RAND_MAX));
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
	if (settings->isConstrainedProblem()) {
		for (unsigned int j = 0; j < number_of_experiments; j++) {
			buffer[j].resize(n);
		}
	}

	double start_time_of_iterations = gettime();
	for (unsigned int it = 0; it < settings->max_it; it++) {
		for (unsigned int tmp = 0; tmp < TOTAL_THREADS; tmp++) {
			max_errors[tmp] = 0;
		}
		if (settings->isConstrainedProblem()) {

			if (doMean) {
				my_mm_multiply(false, m, n, number_of_experiments, B_CSC_Vals,
						B_CSC_Row_Id, B_CSC_Col_Ptr, means, V, Z);
			} else {
				sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_NOTRANS, m,
						number_of_experiments, n, &floating_one, matdescra,
						B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr,
						&B_CSC_Col_Ptr[1], V, ONE_MKL_INT, &floating_zero, Z,
						ONE_MKL_INT);
			}

			//set Z=sgn(Z)
			if (settings->algorithm == L0_constrained_L1_PCA
					|| settings->algorithm == L1_constrained_L1_PCA) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (unsigned int j = 0; j < number_of_experiments; j++) {
					vals[j].tmp = cblas_l1_norm(m, &Z[m * j], 1);
					vector_sgn(&Z[m * j], m);//y=sgn(y)
				}
			}

			if (doMean) {
				my_mm_multiply(true, m, n, number_of_experiments, B_CSC_Vals,
						B_CSC_Row_Id, B_CSC_Col_Ptr, means, Z, V);
			} else {

				sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_TRANS, m,
						number_of_experiments, n, &floating_one, matdescra,
						B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr,
						&B_CSC_Col_Ptr[1], Z, ONE_MKL_INT, &floating_zero, V,
						ONE_MKL_INT);
			}
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (unsigned int j = 0; j < number_of_experiments; j++) {
				F fval_current = 0;
				if (settings->algorithm == L0_constrained_L2_PCA
						|| settings->algorithm == L1_constrained_L2_PCA) {
					fval_current = cblas_l2_norm(m, &Z[m * j], 1);
				}
				F norm_of_x;
				if (settings->isL1ConstrainedProblem()) {
					norm_of_x = soft_tresholding(&V[n * j], n,
							settings->constrain, buffer[j]); // x = S_w(x)
				} else {
					norm_of_x = k_hard_tresholding(&V[n * j], n,
							settings->constrain, buffer[j], settings); // x = T_k(x)
				}

				cblas_vector_scale(n, &V[j * n], 1 / norm_of_x);
				if (settings->algorithm == L0_constrained_L1_PCA
						|| settings->algorithm == L1_constrained_L1_PCA) {
					fval_current = vals[j].tmp;
				}
				F tmp_error = computeTheError(fval_current, vals[j].val,settings);
				//Log end of iteration for given point
				if (settings->get_it_for_all_points && termination_criteria(
						tmp_error, it, settings) && stat->iters[j] == -1) {
					stat->iters[j] = it;
				} else if (settings->get_it_for_all_points
						&& !termination_criteria(tmp_error, it, settings)
						&& stat->iters[j] != -1) {
					stat->iters[j] = -1;
				}
				//---------------
				if (max_errors[my_thread_id] < tmp_error)
					max_errors[my_thread_id] = tmp_error;
				vals[j].val = fval_current;
			}
		} else {
			//scale Z
			if (settings->algorithm == L0_penalized_L1_PCA
					|| settings->algorithm == L1_penalized_L1_PCA) {
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

			if (settings->isL1PenalizedProblem()) {
				L1_penalized_tresholding(number_of_experiments, n, V, settings,
						max_errors, vals, stat, it);
			} else {
				L0_penalized_tresholding(number_of_experiments, n, V, settings,
						max_errors, vals, stat, it);
			}

			sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_NOTRANS, m,
					number_of_experiments, n, &floating_one, matdescra,
					B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr, &B_CSC_Col_Ptr[1],
					V, ONE_MKL_INT, &floating_zero, Z, ONE_MKL_INT);
		}
		error
				= max_errors[cblas_vector_max_index(TOTAL_THREADS, max_errors,
						1)];
		if (termination_criteria(error, it, settings)) {
			stat->it = it;
			break;
		}

	}
	double end_time_of_iterations = gettime();
	//compute corresponding x
	stat->values.resize(settings->starting_points);
	int selected_idx = 0;
	F best_value = vals[selected_idx].val;
	stat->values[0] = best_value;
	stat->true_computation_time = (end_time_of_iterations
			- start_time_of_iterations);
	for (unsigned int i = 1; i < number_of_experiments; i++) {
		stat->values[i] = vals[i].val;
		if (vals[i].val > best_value) {
			best_value = vals[i].val;
			selected_idx = i;
		}
	}
	cblas_vector_copy(n, &V[n * selected_idx], 1, x, 1);
	F norm_of_x = cblas_l2_norm(n, x, 1);
	cblas_vector_scale(n, x, 1 / norm_of_x);//Final x
	free(Z);
	free(V);
	free(vals);
	return best_value;
}

#endif /* SPARSE_PCA_SOLVER_H__ */
