/*
 * l0_penalized_l2_PCA.h
 *
 *  Created on: Mar 26, 2012
 *      Author: taki
 *
 *  min  max_{\|x\|_2 \leq1}  \|Bx\|_2^2 - \gamma \|x\|_0
 *  from paper: XXXXXXXXXXXX
 *
 */

#ifndef SPARSE_PCA_SOLVER_H_
#define SPARSE_PCA_SOLVER_H_
#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
#include "../utils/various.h"
#include "../utils/tresh_functions.h"
#include "../utils/timer.h"
#include "sparse_PCA_tresholding.h"

#include "../utils/gsl_helper.h"

#include "../utils/my_cblas_wrapper.h"


#include "gpower_commons.h"

/*
 * Matrix B is stored in column order (Fortran Based)
 */




template<typename F>
F sparse_PCA_solver(const F * B, const int ldB, F * x, const unsigned int m,
		const unsigned int n, optimization_settings* settings,
		optimization_statistics* stat) {
	stat->it = settings-> max_it;
	F FLOATING_ZERO = 0;
	if (settings->batch_size==0){
		settings->batch_size=settings->starting_points;
	}
	int number_of_batches = settings->starting_points / settings->batch_size;
	if (number_of_batches * settings->batch_size < settings->starting_points)
		number_of_batches++;
	settings->starting_points = number_of_batches * settings->batch_size;

	// Allocate vector for stat to return which point needs how much iterations
	if (settings->get_it_for_all_points) {
		stat->iters.resize(settings->starting_points, -1);
		stat->cardinalities.resize(settings->starting_points, -1);
		stat->values.resize(settings->starting_points, -1);

	}
	const unsigned int number_of_experiments_per_batch = settings->batch_size;
	F * Z = (F*) calloc(m * number_of_experiments_per_batch, sizeof(F));
	value_coordinate_holder<F>* vals = (value_coordinate_holder<F>*) calloc(
			number_of_experiments_per_batch,
			sizeof(value_coordinate_holder<F> ));
	F * V = (F*) calloc(n * number_of_experiments_per_batch, sizeof(F));
	stat->true_computation_time = 0;
	F error = 0;
	F max_errors[TOTAL_THREADS];
	std::vector<F>* buffer = (std::vector<F>*) calloc(
			number_of_experiments_per_batch, sizeof(std::vector<F>));
	if (settings->isConstrainedProblem()) {
		for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
			buffer[j].resize(n);
		}
	}
	F the_best_solution_value = -1;
	unsigned int total_iterations = 0;
	stat->it = 0;
	if (settings->on_the_fly_generation) {
		settings->get_it_for_all_points = false;
		cblas_vector_scale(n * number_of_experiments_per_batch, V, FLOATING_ZERO);
		initialize_starting_points(V, Z, settings, stat,
				number_of_experiments_per_batch, n, m, ldB, B, 0);
		unsigned int generated_points = number_of_experiments_per_batch;
		bool do_iterate = true;
		unsigned int statistical_shift = 0;

		std::vector<unsigned int> current_iteration(
				number_of_experiments_per_batch, 0);
		std::vector<unsigned int> current_order(
				number_of_experiments_per_batch, 0);
		for (unsigned int i = 0; i < number_of_experiments_per_batch; i++) {
			current_order[i] = i;
		}
		while (do_iterate) {
			total_iterations++;
			for (unsigned int tmp = 0; tmp < TOTAL_THREADS; tmp++) {
				max_errors[tmp] = 0;
			}
			if (settings->isConstrainedProblem()) {
				perform_one_iteration_for_constrained_pca(V, Z, settings, stat,
						number_of_experiments_per_batch, n, m, ldB, B,
						max_errors, vals, buffer, 0, statistical_shift);
			} else {
				perform_one_iteration_for_penalized_pca(V, Z, settings, stat,
						number_of_experiments_per_batch, n, m, ldB, B,
						max_errors, vals, 0, statistical_shift);
			}

			do_iterate = false;
			unsigned int number_of_new_points = 0;
			for (unsigned int i = 0; i < number_of_experiments_per_batch; i++) {
				current_iteration[i]++;
				if (termination_criteria(vals[i].current_error,
						current_iteration[i], settings) || current_iteration[i]
						>= settings->max_it) {
					// this point reached it convergence criterion, stat again....
					if (the_best_solution_value < vals[i].val) {
						the_best_solution_value = vals[i].val;
						cblas_vector_copy(n, &V[n * i], 1, x, 1);
					}
					if (generated_points < settings->starting_points) {
						vals[i].reset();
						current_order[i] = generated_points;
						number_of_new_points++;
						generated_points++;
						current_iteration[i] = 0;
						do_iterate = true;
					}
				} else {
					do_iterate = true;
				}
			}
			if (number_of_new_points > 0) {
				// generate new points in parallel
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
					if (current_iteration[j] == 0) {
						if (settings->isConstrainedProblem()) {
							cblas_vector_scale(n, &V[j * n], FLOATING_ZERO);
						}
						getSignleStartingPoint(&V[j * n], &Z[j * m], settings,
								n, m, current_order[j], 0);
					}
				}
			}

		}
	} else {
		//====================== MAIN LOOP THROUGHT BATCHES
		for (unsigned int batch = 0; batch < number_of_batches; batch++) {
			unsigned int statistical_shift = batch * settings->batch_size;
			cblas_vector_scale(n * number_of_experiments_per_batch, V, FLOATING_ZERO);
			initialize_starting_points(V, Z, settings, stat,
					number_of_experiments_per_batch, n, m, ldB, B,
					statistical_shift);
			for (int j = 0; j < number_of_experiments_per_batch; j++) {
				vals[j].reset();
			}
			double start_time_of_iterations = gettime();
			for (unsigned int it = 0; it < settings->max_it; it++) {
				total_iterations++;
				for (unsigned int tmp = 0; tmp < TOTAL_THREADS; tmp++) {
					max_errors[tmp] = 0;
				}
				if (settings->isConstrainedProblem()) {
					perform_one_iteration_for_constrained_pca(V, Z, settings,
							stat, number_of_experiments_per_batch, n, m, ldB,
							B, max_errors, vals, buffer, it, statistical_shift);
				} else {
					perform_one_iteration_for_penalized_pca(V, Z, settings,
							stat, number_of_experiments_per_batch, n, m, ldB,
							B, max_errors, vals, it, statistical_shift);
				}
				error = max_errors[cblas_vector_max_index(TOTAL_THREADS,
						max_errors, 1)];
				if (termination_criteria(error, it, settings)) {
					break;
				}
			}
			double end_time_of_iterations = gettime();
			stat->true_computation_time += (end_time_of_iterations
					- start_time_of_iterations);
			//============= save the best solution==========
			int selected_idx = 0;
			F best_value = vals[selected_idx].val;
			if (settings->get_it_for_all_points)
				stat->values[0 + statistical_shift] = best_value;
			for (unsigned int i = 1; i < number_of_experiments_per_batch; i++) {
				if (settings->get_it_for_all_points)
					stat->values[i + statistical_shift] = vals[i].val;
				if (vals[i].val > best_value) {
					best_value = vals[i].val;
					selected_idx = i;
				}
			}
			if (the_best_solution_value < best_value) { //if this batch gives better value, store "x"
				the_best_solution_value = best_value;
				cblas_vector_copy(n, &V[n * selected_idx], 1, x, 1);
			}
		}
	}
	stat->it = total_iterations;
	//compute corresponding x
	F norm_of_x = cblas_l2_norm(n, x, 1);
	cblas_vector_scale(n, x, 1 / norm_of_x);//Final x
	free(Z);
	free(V);
	free(vals);
	return the_best_solution_value;
}




#endif /* SPARSE_PCA_SOLVER_H__ */
//		const gsl_rng_type * T;
//		gsl_rng * r;
//		gsl_rng_env_setup();
//		T = gsl_rng_default;
//		r = gsl_rng_alloc(T);
//				F tmp = gsl_ran_ugaussian (r);
