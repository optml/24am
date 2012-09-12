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

#ifndef GPU_SPARSE_PCA_SOLVER_SINGLE_H_
#define GPU_SPARSE_PCA_SOLVER_SINGLE_H_

#include "../gpower/optimization_settings.h"
#include <cuda.h>
//#include "cublas.h"
#include "cublas_v2.h"

#include "../gpower/termination_criteria.h"
#include "my_cublas_wrapper.h"
#include <thrust/sort.h>
#include <thrust/functional.h>

/*
 * POPIS TODO
 */
template<typename F>
int gpu_sparse_PCA_solver_single(cublasHandle_t &handle,const unsigned int m, const unsigned int n,
		thrust::device_vector<F> d_B, thrust::host_vector<F>& h_x,
		optimization_settings* settings, optimization_statistics* stat) {
	const F penalty = (F) settings->penalty;
	//initialize cublas
//	cublasStatus_t status;
//	cublasHandle_t handle;
//	status = cublasCreate(&handle);
//	if (status != CUBLAS_STATUS_SUCCESS) {
//		fprintf(stderr, "! CUBLAS initialization error\n");
//		return EXIT_FAILURE;
//	} else {
//		printf("CUBLAS initialized.\n");
//	}

	thrust::host_vector<F> h_z(m, 1);
	//Initialize random starting vector from [-1,1]^m
	//	thrust::generate(h_z.begin(), h_z.end(), rand);//FIXME FAKE
	thrust::device_vector<F> d_z = h_z;
	//alocate vector "x" on device
	thrust::device_vector<F> d_x(n, 0);
	thrust::device_vector<F> d_x_for_sort;
	if (settings->isConstrainedProblem()) {
		d_x_for_sort.resize(n);
		for (int i=0;i<settings->constrain;i++){
			d_x[i]=1;
		}

		//		thrust::generate(d_x.begin(), d_x.end(), rand);
	}

	// create raw pointers
	F * z = thrust::raw_pointer_cast(&d_z[0]);
	F * x = thrust::raw_pointer_cast(&d_x[0]);
	F * B = thrust::raw_pointer_cast(&d_B[0]);
	F scale_factor = 1;
	F ONE = 1;
	F norm_of_z = 0;
			F norm_of_x = 0;
	F fval = 0;
	F fval_prev = 0;
	stat->it = settings-> max_it;
	for (unsigned int it = 0; it < settings->max_it; it++) {

		if (settings->isConstrainedProblem()) {
			//=================CONSTRAINED METHODS
			gpu_computeNorm(handle, x, n, norm_of_x);
			scale_factor=1/norm_of_x;
			// Multiply z = B*x
			gpu_matrix_vector_multiply(handle, CUBLAS_OP_N, scale_factor, m, n, B, x, z);
			//set Z=sgn(Z)
			if (settings->algorithm == L0_constrained_L1_PCA
					|| settings->algorithm == L1_constrained_L1_PCA) {
				gpu_compute_l1_Norm(handle, z, m, fval);
				thrust::transform(d_z.begin(), d_z.end(), d_z.begin(),
						gpu_sgn_transformator<F> ());
			}
			// Multiply x = B'*z
			gpu_matrix_vector_multiply(handle, CUBLAS_OP_T, ONE, m, n, B, z, x);
			if (settings->algorithm == L0_constrained_L2_PCA
					|| settings->algorithm == L1_constrained_L2_PCA) {
				gpu_computeNorm(handle, z, m, fval);
			}
			// copy data to x_for_sort
			thrust::copy(d_x.begin(), d_x.end(), d_x_for_sort.begin());
			// sort data on the device
			thrust::sort(d_x_for_sort.begin(), d_x_for_sort.end(),
					comparator_abs_val<F>());// maybe use stable_sort ???
			if (settings->isL1ConstrainedProblem()) {
				thrust::copy(d_x_for_sort.begin(), d_x_for_sort.end(),
						h_x.begin());
				F tresh_hold = compute_soft_tresholding_parameter(&h_x[0], n,
						settings->constrain);
				thrust::transform(d_x.begin(), d_x.end(), d_x.begin(),
						gpu_soft_treshholding<F> (tresh_hold));
			} else {
				F tresh_hold = abs(d_x_for_sort[settings->constrain - 1]);
				thrust::transform(d_x.begin(), d_x.end(), d_x.begin(),
						gpu_hard_treshholding<F> (tresh_hold));
			}
		} else {
			//=================PENALIZED METHODS
			if (settings->algorithm == L0_penalized_L1_PCA
					|| settings->algorithm == L1_penalized_L1_PCA) {
				//z=sgn(z)
				thrust::transform(d_z.begin(), d_z.end(), d_z.begin(),
						gpu_sgn_transformator<F> ());
			} else {
				gpu_computeNorm(handle, z, m, norm_of_z);
				scale_factor = 1 / norm_of_z;
			}
			// Multiply x = B'*z*scale_factor
			gpu_matrix_vector_multiply(handle, CUBLAS_OP_T, scale_factor, m, n,
					B, z, x);
			if (settings->isL1PenalizedProblem()) {
				//------------L1 PENALIZED
				fval = thrust::transform_reduce(d_x.begin(), d_x.end(),
						gpu_l1_penalized_objval<F> (penalty), 0.0f,
						thrust::plus<F>());
				thrust::transform(d_x.begin(), d_x.end(), d_x.begin(),
						gpu_l1_penalized_tresholing<F> (penalty));
				fval = std::sqrt(fval);
			} else {
				//------------L0 PENALIZED
				fval = thrust::transform_reduce(d_x.begin(), d_x.end(),
						gpu_l0_penalized_objval<F> (penalty), 0.0f,
						thrust::plus<F>());
				thrust::transform(d_x.begin(), d_x.end(), d_x.begin(),
						gpu_l0_penalized_tresholing<F> (penalty));
			}
			// Multiply z = B*x
			gpu_matrix_vector_multiply(handle, CUBLAS_OP_N, ONE, m, n, B, x, z);


		}

		if (termination_criteria(fval, fval_prev, it, settings)) {
			stat->it = it;
			break;
		}
		fval_prev = fval;
	}
	stat->fval = fval;
	// compute corresponding x
	gpu_computeNorm(handle, x, n, norm_of_x);
	//Final x (scale so the norm is one
	gpu_vector_scale(handle, x, n, 1 / norm_of_x);
	// copy from device to host
	thrust::copy(d_x.begin(), d_x.end(), h_x.begin());
	// destoy cublas
//	status = cublasDestroy(handle);
//		if (status != CUBLAS_STATUS_SUCCESS) {
//			fprintf(stderr, "!cublas shutdown error\n");
//			return EXIT_FAILURE;
//		}
	return 0;
}

#endif /* GPU_SPARSE_PCA_SOLVER_H__ */
