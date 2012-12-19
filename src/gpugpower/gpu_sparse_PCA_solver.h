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

#ifndef denseDataSolver_H_
#define denseDataSolver_H_

#define MEMORY_ALIGNMENT 128

#include "../class/optimization_settings.h"
#include "../utils/termination_criteria.h"
#include "gpu_headers.h"

namespace SPCASolver {
namespace GPUSolver{
/*
 * POPIS TODO
 */
template<typename F>
int denseDataSolver(cublasHandle_t &handle, const unsigned int m,
		const unsigned int n, thrust::device_vector<F> &d_B,
		thrust::host_vector<F>& h_x,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics,
		const unsigned int LD_M, const unsigned int LD_N) {
	bool low_memory = false;
	double total_memory = LD_N * LD_M + LD_N * optimizationSettings->totalStartingPoints * 2
			+ LD_M * optimizationSettings->totalStartingPoints;
	total_memory = total_memory * sizeof(F)
			+ LD_N * optimizationSettings->totalStartingPoints * 4;
	total_memory = total_memory / (1024 * 1024 * 1024);
	if (optimizationSettings->verbose) {
		printf("Total Memory needed = %f GB\n", total_memory);
	}
	if (total_memory > 1) {
		low_memory = true;
	}

	//Initialize random starting vector from [-1,1]^m
	thrust::device_vector<F> d_z(LD_M * optimizationSettings->totalStartingPoints, 0);
	if (!optimizationSettings->isConstrainedProblem()) {
		generate_random_number<F> generator(1);
		thrust::transform(thrust::make_counting_iterator < F > (0),
				thrust::make_counting_iterator < F
						> (LD_M * optimizationSettings->totalStartingPoints), d_z.begin(),
				generator);
	}
	//alocate vector "x" on device
	thrust::device_vector<F> d_x(n, 0);
	thrust::device_vector<F> d_V(LD_N * optimizationSettings->totalStartingPoints, 0);
	thrust::device_vector<F> d_x_for_sort;
	thrust::device_vector<int> d_IDX;
	thrust::device_vector<F> dataToSort;
	if (optimizationSettings->isConstrainedProblem()) {
		if (low_memory) {
			d_x_for_sort.resize(LD_N);
		} else {
			d_IDX.resize(optimizationSettings->totalStartingPoints * LD_N, 0);
			thrust::transform(thrust::make_counting_iterator < F > (0),
					thrust::make_counting_iterator < F
							> (LD_N * optimizationSettings->totalStartingPoints), d_IDX.begin(),
					init_sequence_with_LDN(LD_N));
			dataToSort.resize(LD_N * optimizationSettings->totalStartingPoints);
		}
		for (int i = 0; i < optimizationSettings->constrain; i++) {
			for (int j = 0; j < optimizationSettings->totalStartingPoints; j++) {
				int tmp_idx = n * (rand() / (RAND_MAX + 0.0));
				tmp_idx = (tmp_idx == n ? tmp_idx-- : tmp_idx);
				d_V[tmp_idx + j * LD_N] = rand() / (0.0 + RAND_MAX);
			}
		}
	}
	// create raw pointers
	F * z = thrust::raw_pointer_cast(&d_z[0]);
	F * V = thrust::raw_pointer_cast(&d_V[0]);
	F * x = thrust::raw_pointer_cast(&d_x[0]);
	F * B = thrust::raw_pointer_cast(&d_B[0]);
	F ONE = 1;
	F norm_of_z = 0;
	F norm_of_x = 0;
	optimizationStatistics->it = optimizationSettings->maximumIterations;
	F error = 0;
	ValueCoordinateHolder<F>* vals = (ValueCoordinateHolder<F>*) calloc(
			optimizationSettings->totalStartingPoints, sizeof(ValueCoordinateHolder<F> ));
	//-------measuere time of main LOOP
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	/* -----------------------START ITERATING -------------*/
	for (unsigned int it = 0; it < optimizationSettings->maximumIterations; it++) {
		if (optimizationSettings->isConstrainedProblem()) {
			//=================CONSTRAINED METHODS
			for (unsigned int i = 0; i < optimizationSettings->totalStartingPoints; i++) {
				gpu_computeNorm(handle, &V[i * LD_N], n, norm_of_x);
				gpu_vector_scale(handle, &V[i * LD_N], n, 1 / norm_of_x);
								gpu_computeNorm(handle, &V[i * LD_N], n, norm_of_x);
			}
			// Multiply z = B*x
			gpu_matrix_matrix_multiply(handle, CUBLAS_OP_N, ONE, m, n, B, V, z,
					optimizationSettings->totalStartingPoints, LD_M, LD_N);

			//set Z=sgn(Z)
			if (optimizationSettings->algorithm == L0_constrained_L1_PCA
					|| optimizationSettings->algorithm == L1_constrained_L1_PCA) {
				for (unsigned int i = 0; i < optimizationSettings->totalStartingPoints; i++) {
					gpu_compute_l1_Norm(handle, &z[LD_M * i], m, vals[i].val);
				}
				thrust::transform(d_z.begin(), d_z.end(), d_z.begin(),
						gpu_sgn_transformator<F>());
			}

			// Multiply x = B'*z
			gpu_matrix_matrix_multiply(handle, CUBLAS_OP_T, ONE, m, n, B, z, V,
					optimizationSettings->totalStartingPoints, LD_M, LD_N);

			if (optimizationSettings->algorithm == L0_constrained_L2_PCA
					|| optimizationSettings->algorithm == L1_constrained_L2_PCA) {
				for (unsigned int i = 0; i < optimizationSettings->totalStartingPoints; i++) {
					gpu_computeNorm(handle, &z[LD_M * i], m, vals[i].val);
				}
			}
			//			cudaEvent_t start2, stop2;
			//					float time2;
			//					cudaEventCreate(&start2);
			//					cudaEventCreate(&stop2);
			//					cudaEventRecord(start2, 0);
			// choose betweem those two implementation
			if (low_memory) {
				perform_hard_and_soft_thresholding(d_V, optimizationSettings, n,
						d_x_for_sort, h_x, LD_N);
			} else {
				perform_hard_and_soft_thresholdingNEW(d_V, optimizationSettings, n, h_x,
						LD_N, d_IDX, dataToSort);
			}
			//			cudaEventRecord(stop2, 0);
			//								cudaEventSynchronize(stop2);
			//								cudaEventElapsedTime(&time2, start2, stop2);
			//								cudaEventDestroy(start2);
			//								cudaEventDestroy(stop2);
			//								printf("thresholign %f\n", time2);
		} else {
			//=================PENALIZED METHODS
			if (optimizationSettings->algorithm == L0_penalized_L1_PCA
					|| optimizationSettings->algorithm == L1_penalized_L1_PCA) {
				//z=sgn(z)
				thrust::transform(d_z.begin(), d_z.end(), d_z.begin(),
						gpu_sgn_transformator<F>());
			} else {
				for (unsigned int i = 0; i < optimizationSettings->totalStartingPoints; i++) {
					gpu_computeNorm(handle, &z[i * LD_M], m, norm_of_z);
					gpu_vector_scale(handle, &z[i * LD_M], m, 1 / norm_of_z);
				}
			}

			// Multiply x = B'*z
			gpu_matrix_matrix_multiply(handle, CUBLAS_OP_T, ONE, m, n, B, z, V,
					optimizationSettings->totalStartingPoints, LD_M, LD_N);
			perform_hard_and_soft_thresholding_for_penalized(d_V, optimizationSettings, n,
					vals, LD_N);
			// Multiply z = B*x
			gpu_matrix_matrix_multiply(handle, CUBLAS_OP_N, ONE, m, n, B, V, z,
					optimizationSettings->totalStartingPoints, LD_M, LD_N);
		}
		error = 0;
		for (unsigned int i = 0; i < optimizationSettings->totalStartingPoints; i++) {
			F tmp = computeTheError(vals[i].val, vals[i].prev_val, optimizationSettings);
			if (tmp > error)
				error = tmp;
			vals[i].prev_val = vals[i].val;
		}
		if (termination_criteria(error, it, optimizationSettings)) {
			optimizationStatistics->it = it;
			break;
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	optimizationStatistics->totalTrueComputationTime = time;
	int selected_idx = 0;
	F best_value = vals[selected_idx].val;
	for (unsigned int i = 1; i < optimizationSettings->totalStartingPoints; i++) {
		if (vals[i].val > best_value) {
			best_value = vals[i].val;
			selected_idx = i;
		}
	}
	gpu_vector_copy(handle, &V[LD_N * selected_idx], n, x);
	optimizationStatistics->fval = best_value;
	// compute corresponding x
	gpu_computeNorm(handle, x, n, norm_of_x);
	//Final x (scale so the norm is one
	gpu_vector_scale(handle, x, n, 1 / norm_of_x);
	// copy from device to host
	thrust::copy(d_x.begin(), d_x.end(), h_x.begin());
	return 0;
}
}
}
#endif /* denseDataSolver_H__ */
