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

#ifndef SPARSE_PCA_SOLVER_H_
#define SPARSE_PCA_SOLVER_H_
#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
#include "../utils/various.h"
#include "../utils/thresh_functions.h"
#include "../utils/timer.h"
#include "sparse_PCA_thresholding.h"

#include "../utils/gsl_helper.h"

#include "../utils/my_cblas_wrapper.h"

#include "gpower_commons.h"

/*
 * Matrix B is stored in column order (Fortran Based)
 */

namespace SPCASolver {
namespace MulticoreSolver{
template<typename F>
F denseDataSolver(const F * B, const int ldB, F * x, const unsigned int m,
		const unsigned int n, SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics) {
#ifdef _OPENMP
#pragma omp parallel
	{
	optimizationStatistics->totalThreadsUsed = omp_get_num_threads();
	}
#endif

	if (optimizationSettings->verbose) {
		cout << "Solver started " << endl;
	}
	optimizationSettings->chceckInputAndModifyIt(n);
	optimizationStatistics->it = optimizationSettings->maximumIterations;
	F FLOATING_ZERO = 0;
	// Allocate vector for optimizationStatistics to return which point needs how much iterations
	if (optimizationSettings->storeIterationsForAllPoints) {
		optimizationStatistics->iters.resize(optimizationSettings->totalStartingPoints, -1);
		optimizationStatistics->cardinalities.resize(optimizationSettings->totalStartingPoints, -1);
		optimizationStatistics->values.resize(optimizationSettings->totalStartingPoints, -1);

	}
	const unsigned int number_of_experiments_per_batch = optimizationSettings->batchSize;
	F * Z = (F*) calloc(m * number_of_experiments_per_batch, sizeof(F));
	ValueCoordinateHolder<F>* vals = (ValueCoordinateHolder<F>*) calloc(
			number_of_experiments_per_batch,
			sizeof(ValueCoordinateHolder<F> ));
	F * V = (F*) calloc(n * number_of_experiments_per_batch, sizeof(F));
	optimizationStatistics->totalTrueComputationTime = 0;
	F error = 0;
	F max_errors[TOTAL_THREADS];
	std::vector<F>* buffer = (std::vector<F>*) calloc(
			number_of_experiments_per_batch, sizeof(std::vector<F>));
	if (optimizationSettings->isConstrainedProblem()) {
		for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
			buffer[j].resize(n);
		}
	}
	F the_best_solution_value = -1;
	unsigned int total_iterations = 0;
	optimizationStatistics->it = 0;
	if (optimizationSettings->onTheFlyMethod) {
		optimizationSettings->storeIterationsForAllPoints = false;
		cblas_vector_scale(n * number_of_experiments_per_batch, V,
				FLOATING_ZERO);
		initialize_totalStartingPoints(V, Z, optimizationSettings, optimizationStatistics,
				number_of_experiments_per_batch, n, m, ldB, B, 0);
		unsigned int generated_points = number_of_experiments_per_batch;
		bool do_iterate = true;
		unsigned int optimizationStatisticsistical_shift = 0;

		std::vector<unsigned int> current_iteration(
				number_of_experiments_per_batch, 0);
		std::vector<unsigned int> current_order(number_of_experiments_per_batch,
				0);
		for (unsigned int i = 0; i < number_of_experiments_per_batch; i++) {
			current_order[i] = i;
		}
		double start_time_of_iterations = gettime();
		while (do_iterate) {
			total_iterations++;
			for (unsigned int tmp = 0; tmp < TOTAL_THREADS; tmp++) {
				max_errors[tmp] = 0;
			}
			if (optimizationSettings->isConstrainedProblem()) {
				perform_one_iteration_for_constrained_pca(V, Z, optimizationSettings, optimizationStatistics,
						number_of_experiments_per_batch, n, m, ldB, B,
						max_errors, vals, buffer, 0, optimizationStatisticsistical_shift);
			} else {
				perform_one_iteration_for_penalized_pca(V, Z, optimizationSettings, optimizationStatistics,
						number_of_experiments_per_batch, n, m, ldB, B,
						max_errors, vals, 0, optimizationStatisticsistical_shift);
			}

			do_iterate = false;
			unsigned int number_of_new_points = 0;
			for (unsigned int i = 0; i < number_of_experiments_per_batch; i++) {
				current_iteration[i]++;
				if (termination_criteria(vals[i].current_error,
						current_iteration[i], optimizationSettings)
						|| current_iteration[i] >= optimizationSettings->maximumIterations) {
					// this point reached it convergence criterion, optimizationStatistics again....
					if (the_best_solution_value < vals[i].val) {
						the_best_solution_value = vals[i].val;
						cblas_vector_copy(n, &V[n * i], 1, x, 1);
					}
					if (generated_points < optimizationSettings->totalStartingPoints) {
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
				for (unsigned int j = 0; j < number_of_experiments_per_batch;
						j++) {
					if (current_iteration[j] == 0) {
						if (optimizationSettings->isConstrainedProblem()) {
							cblas_vector_scale(n, &V[j * n], FLOATING_ZERO);
						}
						getSignleStartingPoint(&V[j * n], &Z[j * m], optimizationSettings,
								n, m, current_order[j], 0);
					}
				}
			}

		}
		double end_time_of_iterations = gettime();
					optimizationStatistics->totalTrueComputationTime += (end_time_of_iterations
							- start_time_of_iterations);
	} else {
		//====================== MAIN LOOP THROUGHT BATCHES
		for (unsigned int batch = 0; batch < optimizationSettings->totalBatches;
				batch++) {
			unsigned int optimizationStatisticsistical_shift = batch * optimizationSettings->batchSize;
			cblas_vector_scale(n * number_of_experiments_per_batch, V,
					FLOATING_ZERO);
			initialize_totalStartingPoints(V, Z, optimizationSettings, optimizationStatistics,
					number_of_experiments_per_batch, n, m, ldB, B,
					optimizationStatisticsistical_shift);
			for (unsigned int j = 0; j < number_of_experiments_per_batch; j++) {
				vals[j].reset();
			}
			double start_time_of_iterations = gettime();
			for (unsigned int it = 0; it < optimizationSettings->maximumIterations; it++) {
				total_iterations++;
				for (unsigned int tmp = 0; tmp < TOTAL_THREADS; tmp++) {
					max_errors[tmp] = 0;
				}
				if (optimizationSettings->isConstrainedProblem()) {
					perform_one_iteration_for_constrained_pca(V, Z, optimizationSettings,
							optimizationStatistics, number_of_experiments_per_batch, n, m, ldB, B,
							max_errors, vals, buffer, it, optimizationStatisticsistical_shift);
				} else {
					perform_one_iteration_for_penalized_pca(V, Z, optimizationSettings,
							optimizationStatistics, number_of_experiments_per_batch, n, m, ldB, B,
							max_errors, vals, it, optimizationStatisticsistical_shift);
				}
				error = max_errors[cblas_vector_max_index(TOTAL_THREADS,
						max_errors, 1)];
				if (termination_criteria(error, it, optimizationSettings)) {
					break;
				}
			}
			double end_time_of_iterations = gettime();
			optimizationStatistics->totalTrueComputationTime += (end_time_of_iterations
					- start_time_of_iterations);
			//============= save the best solution==========
			int selected_idx = 0;
			F best_value = vals[selected_idx].val;
			if (optimizationSettings->storeIterationsForAllPoints)
				optimizationStatistics->values[0 + optimizationStatisticsistical_shift] = best_value;
			for (unsigned int i = 1; i < number_of_experiments_per_batch; i++) {
				if (optimizationSettings->storeIterationsForAllPoints)
					optimizationStatistics->values[i + optimizationStatisticsistical_shift] = vals[i].val;
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
	optimizationStatistics->it = total_iterations;
	//compute corresponding x
	F norm_of_x = cblas_l2_norm(n, x, 1);
	cblas_vector_scale(n, x, 1 / norm_of_x); //Final x
	free(Z);
	free(V);
	free(vals);
	optimizationStatistics->fval = the_best_solution_value;
	return the_best_solution_value;
}
}
}

#endif /* SPARSE_PCA_SOLVER_H__ */
//		const gsl_rng_type * T;
//		gsl_rng * r;
//		gsl_rng_env_setup();
//		T = gsl_rng_default;
//		r = gsl_rng_alloc(T);
//				F tmp = gsl_ran_ugaussian (r);
