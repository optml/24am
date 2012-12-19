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
 *
 *  This file contains a distributed thresholding functions
 *
 */

#ifndef DISTRIBUTED_THRESHOLDINS_H_
#define DISTRIBUTED_THRESHOLDINS_H_

#include <stdio.h>
#include <stdlib.h>
#include "../utils/timer.h"
#include "mkl_constants_and_headers.h"
#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
#include "../utils/termination_criteria.h"
#include "../utils/thresh_functions.h"
#include "../utils/various.h"
#include "distributed_classes.h"

namespace SPCASolver {
namespace distributed_thresholdings {

// this function execute one iteration for penalized PCA
template<typename F>
void perform_one_distributed_iteration_for_penalized_pca(
		SPCASolver::DistributedClasses::OptimizationData<F>& optimizationDataInstance,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics) {
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;

	// z= B*V
	pXgemm(&transNo, &transNo, &optimizationDataInstance.params.DIM_M,
			&optimizationSettings->batchSize, &optimizationDataInstance.params.DIM_N, &one,
			&optimizationDataInstance.B[0], &i_one, &i_one,
			optimizationDataInstance.descB, optimizationDataInstance.V, &i_one,
			&i_one, optimizationDataInstance.descV, &zero,
			optimizationDataInstance.Z, &i_one, &i_one,
			optimizationDataInstance.descZ);
	//================== normalize matrix Z
	//scale Z
	if (optimizationSettings->algorithm == SolverStructures::L0_penalized_L1_PCA
			|| optimizationSettings->algorithm == SolverStructures::L1_penalized_L1_PCA) {
		for (int j = 0; j < optimizationDataInstance.nnz_z; j++) {
			optimizationDataInstance.Z[j] = sgn(optimizationDataInstance.Z[j]);
		}
	} else {
		clear_local_vector(optimizationDataInstance.norms, optimizationSettings->batchSize);
		//data are stored in column order
		for (int i = 0; i < optimizationDataInstance.z_nq; i++) {
			F tmp = 0;
			for (int j = 0; j < optimizationDataInstance.z_mp; j++) {
				tmp += optimizationDataInstance.Z[j
						+ i * optimizationDataInstance.z_mp]
						* optimizationDataInstance.Z[j
								+ i * optimizationDataInstance.z_mp];
			}
			optimizationDataInstance.norms[get_column_coordinate(i,
					optimizationDataInstance.params.mycol,
					optimizationDataInstance.params.npcol,
					optimizationDataInstance.params.x_vector_blocking)] = tmp;
		}
        //	sum up + distribute norms of "Z"
		Xgsum2d(&optimizationDataInstance.params.ictxt, &C_CHAR_SCOPE_ALL,
				&C_CHAR_GENERAL_TREE_CATHER, &optimizationSettings->batchSize, &i_one,
				optimizationDataInstance.norms, &optimizationSettings->batchSize, &i_negone,
				&i_negone);
		//normalize local "z"
		for (int i = 0; i < optimizationDataInstance.z_nq; i++) {
			F scaleNorm =
					1
							/ sqrt(
									optimizationDataInstance.norms[get_column_coordinate(
											i,
											optimizationDataInstance.params.mycol,
											optimizationDataInstance.params.npcol,
											optimizationDataInstance.params.x_vector_blocking)]);
			for (int j = 0; j < optimizationDataInstance.z_mp; j++) {
				optimizationDataInstance.Z[j + i * optimizationDataInstance.z_mp] =
						optimizationDataInstance.Z[j
								+ i * optimizationDataInstance.z_mp] * scaleNorm;
			}
		}
	}
	//======================
	// Multiply V = B'*z
	//		sub(C) := alpha*op(sub(A))*op(sub(B)) + beta*sub(C),
	pXgemm(&trans, &transNo, &optimizationDataInstance.params.DIM_N,
			&optimizationSettings->batchSize, &optimizationDataInstance.params.DIM_M, &one,
			&optimizationDataInstance.B[0], &i_one, &i_one,
			optimizationDataInstance.descB, optimizationDataInstance.Z, &i_one,
			&i_one, optimizationDataInstance.descZ, &zero,
			optimizationDataInstance.V, &i_one, &i_one,
			optimizationDataInstance.descV);
	// perform thresh-holding operations and compute objective values
	clear_local_vector(optimizationDataInstance.norms, optimizationSettings->totalStartingPoints); // we use NORMS to store objective values
	if (optimizationSettings->algorithm == SolverStructures::L0_penalized_L1_PCA
			|| optimizationSettings->algorithm == SolverStructures::L0_penalized_L2_PCA) {
		for (int i = 0; i < optimizationDataInstance.V_nq; i++) {
			for (int j = 0; j < optimizationDataInstance.V_mp; j++) {
				const F tmp = optimizationDataInstance.V[j
						+ i * optimizationDataInstance.V_mp];
				F tmp2 = (tmp * tmp - optimizationSettings->penalty);
				if (tmp2 > 0) {
					optimizationDataInstance.norms[get_column_coordinate(i,
							optimizationDataInstance.params.mycol,
							optimizationDataInstance.params.npcol,
							optimizationDataInstance.params.x_vector_blocking)] +=
							tmp2;
				} else {
					optimizationDataInstance.V[j + i * optimizationDataInstance.V_mp] =
							0;
				}
			}
		}
	} else {
		for (int i = 0; i < optimizationDataInstance.V_nq; i++) {
			for (int j = 0; j < optimizationDataInstance.V_mp; j++) {
				const F tmp = optimizationDataInstance.V[j
						+ i * optimizationDataInstance.V_mp];
				F tmp2 = myabs(tmp) - optimizationSettings->penalty;
				if (tmp2 > 0) {
					optimizationDataInstance.norms[get_column_coordinate(i,
							optimizationDataInstance.params.mycol,
							optimizationDataInstance.params.npcol,
							optimizationDataInstance.params.x_vector_blocking)] +=
							tmp2 * tmp2;
					optimizationDataInstance.V[j + i * optimizationDataInstance.V_mp] =
							tmp2 * sgn(tmp);
				} else {
					optimizationDataInstance.V[j + i * optimizationDataInstance.V_mp] =
							0;
				}
			}
		}
	}
}

// this function executes thresholding operations for constrained PCA
template<typename F>
void threshold_V_for_constrained(
		SPCASolver::DistributedClasses::OptimizationData<F>& optimizationDataInstance,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics) {
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;
	//================== Treshhold matrix V
	optimizationDataInstance.initializeDataForConstrainedMethod(optimizationSettings);
	// obtain V from all cluster into V_constr_threshold for sorting and thresholding
	pXgeadd(&transNo, &optimizationDataInstance.params.DIM_N,
			&optimizationSettings->batchSize, &one, optimizationDataInstance.V, &i_one,
			&i_one, optimizationDataInstance.descV, &zero,
			optimizationDataInstance.V_constr_threshold, &i_one, &i_one,
			optimizationDataInstance.descV_threshold);
	//compute thresholding
	if (optimizationDataInstance.V_tr_mp == optimizationDataInstance.params.DIM_N) {
		for (unsigned int j = 0; j < optimizationDataInstance.V_tr_nq; j++) {
			F norm_of_x;
			if (optimizationSettings->isL1ConstrainedProblem()) {
				norm_of_x =
						soft_thresholding(
								&optimizationDataInstance.V_constr_threshold[optimizationDataInstance.params.DIM_N
										* j],
								optimizationDataInstance.params.DIM_N,
								optimizationSettings->constrain,
								optimizationDataInstance.V_constr_sort_buffer[j],
								optimizationSettings); // x = S_w(x)
			} else {
				optimizationSettings->useSortForHardThresholding = true;
				norm_of_x =
						k_hard_thresholding(
								&optimizationDataInstance.V_constr_threshold[optimizationDataInstance.params.DIM_N
										* j],
								optimizationDataInstance.params.DIM_N,
								optimizationSettings->constrain,
								optimizationDataInstance.V_constr_sort_buffer[j],
								optimizationSettings); // x = T_k(x)
			}
			cblas_vector_scale(optimizationDataInstance.params.DIM_N,
					&optimizationDataInstance.V_constr_threshold[optimizationDataInstance.params.DIM_N
							* j], 1 / norm_of_x);
		}
	}
	//return thresholded values
	pXgeadd(&transNo, &optimizationDataInstance.params.DIM_N,
			&optimizationSettings->batchSize, &one,
			optimizationDataInstance.V_constr_threshold, &i_one, &i_one,
			optimizationDataInstance.descV_threshold, &zero,
			optimizationDataInstance.V, &i_one, &i_one,
			optimizationDataInstance.descV);
}

// this function perform one iteration of contrained PCA
template<typename F>
void perform_one_distributed_iteration_for_constrained_pca(
		SPCASolver::DistributedClasses::OptimizationData<F>& optimizationDataInstance,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics) {
	// standard constants used in MKL library
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;
	clear_local_vector(optimizationDataInstance.norms, optimizationSettings->batchSize); // we use NORMS to store objective values
	// z= B*V
	pXgemm(&transNo, &transNo, &optimizationDataInstance.params.DIM_M,
			&optimizationSettings->batchSize, &optimizationDataInstance.params.DIM_N, &one,
			&optimizationDataInstance.B[0], &i_one, &i_one,
			optimizationDataInstance.descB, optimizationDataInstance.V, &i_one,
			&i_one, optimizationDataInstance.descV, &zero,
			optimizationDataInstance.Z, &i_one, &i_one,
			optimizationDataInstance.descZ);
// compute distributed objective values (each computer has only part of the objective value)
	for (int i = 0; i < optimizationDataInstance.z_nq; i++) {
		F tmp = 0;
		for (int j = 0; j < optimizationDataInstance.z_mp; j++) {

			if (optimizationSettings->algorithm == SolverStructures::L0_constrained_L1_PCA
					|| optimizationSettings->algorithm
							== SolverStructures::L1_constrained_L1_PCA) {
				tmp += abs(
						optimizationDataInstance.Z[j
								+ i * optimizationDataInstance.z_mp]);
			} else {
				tmp += optimizationDataInstance.Z[j
						+ i * optimizationDataInstance.z_mp]
						* optimizationDataInstance.Z[j
								+ i * optimizationDataInstance.z_mp];
			}

		}
		optimizationDataInstance.norms[get_column_coordinate(i,
				optimizationDataInstance.params.mycol,
				optimizationDataInstance.params.npcol,
				optimizationDataInstance.params.x_vector_blocking)] = tmp;
	}
	//set Z=sgn(Z)
	if (optimizationSettings->algorithm == SolverStructures::L0_constrained_L1_PCA
			|| optimizationSettings->algorithm
					== SolverStructures::L1_constrained_L1_PCA) {
		vector_sgn(optimizationDataInstance.Z, optimizationDataInstance.nnz_z);	//y=sgn(y)
	}
	// Multiply V = B'*z
	//		sub(C) := alpha*op(sub(A))*op(sub(B)) + beta*sub(C),
	pXgemm(&trans, &transNo, &optimizationDataInstance.params.DIM_N,
			&optimizationSettings->batchSize, &optimizationDataInstance.params.DIM_M, &one,
			&optimizationDataInstance.B[0], &i_one, &i_one,
			optimizationDataInstance.descB, optimizationDataInstance.Z, &i_one,
			&i_one, optimizationDataInstance.descZ, &zero,
			optimizationDataInstance.V, &i_one, &i_one,
			optimizationDataInstance.descV);
	// perform 	threshold operation and compute objective values
	threshold_V_for_constrained(optimizationDataInstance, optimizationSettings, optimizationStatistics);
}

}
}

#endif /* DISTRIBUTED_THRESHOLDINS_H_ */
