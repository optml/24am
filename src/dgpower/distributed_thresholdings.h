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

namespace PCA_solver {
namespace distributed_thresholdings {

// this function execute one iteration for penalized PCA
template<typename F>
void perform_one_distributed_iteration_for_penalized_pca(
		PCA_solver::distributed_classes::optimization_data<F>& optimization_data_inst,
		solver_structures::optimization_settings* settings,
		solver_structures::optimization_statistics* stat) {
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;

	// z= B*V
	pXgemm(&transNo, &transNo, &optimization_data_inst.params.DIM_M,
			&settings->batch_size, &optimization_data_inst.params.DIM_N, &one,
			&optimization_data_inst.B[0], &i_one, &i_one,
			optimization_data_inst.descB, optimization_data_inst.V, &i_one,
			&i_one, optimization_data_inst.descV, &zero,
			optimization_data_inst.Z, &i_one, &i_one,
			optimization_data_inst.descZ);
	//================== normalize matrix Z
	//scale Z
	if (settings->algorithm == solver_structures::L0_penalized_L1_PCA
			|| settings->algorithm == solver_structures::L1_penalized_L1_PCA) {
		for (int j = 0; j < optimization_data_inst.nnz_z; j++) {
			optimization_data_inst.Z[j] = sgn(optimization_data_inst.Z[j]);
		}
	} else {
		clear_local_vector(optimization_data_inst.norms, settings->batch_size);
		//data are stored in column order
		for (int i = 0; i < optimization_data_inst.z_nq; i++) {
			F tmp = 0;
			for (int j = 0; j < optimization_data_inst.z_mp; j++) {
				tmp += optimization_data_inst.Z[j
						+ i * optimization_data_inst.z_mp]
						* optimization_data_inst.Z[j
								+ i * optimization_data_inst.z_mp];
			}
			optimization_data_inst.norms[get_column_coordinate(i,
					optimization_data_inst.params.mycol,
					optimization_data_inst.params.npcol,
					optimization_data_inst.params.x_vector_blocking)] = tmp;
		}
        //	sum up + distribute norms of "Z"
		Xgsum2d(&optimization_data_inst.params.ictxt, &C_CHAR_SCOPE_ALL,
				&C_CHAR_GENERAL_TREE_CATHER, &settings->batch_size, &i_one,
				optimization_data_inst.norms, &settings->batch_size, &i_negone,
				&i_negone);
		//normalize local "z"
		for (int i = 0; i < optimization_data_inst.z_nq; i++) {
			F scaleNorm =
					1
							/ sqrt(
									optimization_data_inst.norms[get_column_coordinate(
											i,
											optimization_data_inst.params.mycol,
											optimization_data_inst.params.npcol,
											optimization_data_inst.params.x_vector_blocking)]);
			for (int j = 0; j < optimization_data_inst.z_mp; j++) {
				optimization_data_inst.Z[j + i * optimization_data_inst.z_mp] =
						optimization_data_inst.Z[j
								+ i * optimization_data_inst.z_mp] * scaleNorm;
			}
		}
	}
	//======================
	// Multiply V = B'*z
	//		sub(C) := alpha*op(sub(A))*op(sub(B)) + beta*sub(C),
	pXgemm(&trans, &transNo, &optimization_data_inst.params.DIM_N,
			&settings->batch_size, &optimization_data_inst.params.DIM_M, &one,
			&optimization_data_inst.B[0], &i_one, &i_one,
			optimization_data_inst.descB, optimization_data_inst.Z, &i_one,
			&i_one, optimization_data_inst.descZ, &zero,
			optimization_data_inst.V, &i_one, &i_one,
			optimization_data_inst.descV);
	// perform thresh-holding operations and compute objective values
	clear_local_vector(optimization_data_inst.norms, settings->starting_points); // we use NORMS to store objective values
	if (settings->algorithm == solver_structures::L0_penalized_L1_PCA
			|| settings->algorithm == solver_structures::L0_penalized_L2_PCA) {
		for (int i = 0; i < optimization_data_inst.V_nq; i++) {
			for (int j = 0; j < optimization_data_inst.V_mp; j++) {
				const F tmp = optimization_data_inst.V[j
						+ i * optimization_data_inst.V_mp];
				F tmp2 = (tmp * tmp - settings->penalty);
				if (tmp2 > 0) {
					optimization_data_inst.norms[get_column_coordinate(i,
							optimization_data_inst.params.mycol,
							optimization_data_inst.params.npcol,
							optimization_data_inst.params.x_vector_blocking)] +=
							tmp2;
				} else {
					optimization_data_inst.V[j + i * optimization_data_inst.V_mp] =
							0;
				}
			}
		}
	} else {
		for (int i = 0; i < optimization_data_inst.V_nq; i++) {
			for (int j = 0; j < optimization_data_inst.V_mp; j++) {
				const F tmp = optimization_data_inst.V[j
						+ i * optimization_data_inst.V_mp];
				F tmp2 = myabs(tmp) - settings->penalty;
				if (tmp2 > 0) {
					optimization_data_inst.norms[get_column_coordinate(i,
							optimization_data_inst.params.mycol,
							optimization_data_inst.params.npcol,
							optimization_data_inst.params.x_vector_blocking)] +=
							tmp2 * tmp2;
					optimization_data_inst.V[j + i * optimization_data_inst.V_mp] =
							tmp2 * sgn(tmp);
				} else {
					optimization_data_inst.V[j + i * optimization_data_inst.V_mp] =
							0;
				}
			}
		}
	}
}

// this function executes thresholding operations for constrained PCA
template<typename F>
void threshold_V_for_constrained(
		PCA_solver::distributed_classes::optimization_data<F>& optimization_data_inst,
		solver_structures::optimization_settings* settings,
		solver_structures::optimization_statistics* stat) {
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;
	//================== Treshhold matrix V
	optimization_data_inst.init_data_for_constrained(settings);
	// obtain V from all cluster into V_constr_threshold for sorting and thresholding
	pXgeadd(&transNo, &optimization_data_inst.params.DIM_N,
			&settings->batch_size, &one, optimization_data_inst.V, &i_one,
			&i_one, optimization_data_inst.descV, &zero,
			optimization_data_inst.V_constr_threshold, &i_one, &i_one,
			optimization_data_inst.descV_threshold);
	//compute thresholding
	if (optimization_data_inst.V_tr_mp == optimization_data_inst.params.DIM_N) {
		for (unsigned int j = 0; j < optimization_data_inst.V_tr_nq; j++) {
			F norm_of_x;
			if (settings->isL1ConstrainedProblem()) {
				norm_of_x =
						soft_thresholding(
								&optimization_data_inst.V_constr_threshold[optimization_data_inst.params.DIM_N
										* j],
								optimization_data_inst.params.DIM_N,
								settings->constrain,
								optimization_data_inst.V_constr_sort_buffer[j],
								settings); // x = S_w(x)
			} else {
				settings->hard_thresholding_using_sort = true;
				norm_of_x =
						k_hard_thresholding(
								&optimization_data_inst.V_constr_threshold[optimization_data_inst.params.DIM_N
										* j],
								optimization_data_inst.params.DIM_N,
								settings->constrain,
								optimization_data_inst.V_constr_sort_buffer[j],
								settings); // x = T_k(x)
			}
			cblas_vector_scale(optimization_data_inst.params.DIM_N,
					&optimization_data_inst.V_constr_threshold[optimization_data_inst.params.DIM_N
							* j], 1 / norm_of_x);
		}
	}
	//return thresholded values
	pXgeadd(&transNo, &optimization_data_inst.params.DIM_N,
			&settings->batch_size, &one,
			optimization_data_inst.V_constr_threshold, &i_one, &i_one,
			optimization_data_inst.descV_threshold, &zero,
			optimization_data_inst.V, &i_one, &i_one,
			optimization_data_inst.descV);
}

// this function perform one iteration of contrained PCA
template<typename F>
void perform_one_distributed_iteration_for_constrained_pca(
		PCA_solver::distributed_classes::optimization_data<F>& optimization_data_inst,
		solver_structures::optimization_settings* settings,
		solver_structures::optimization_statistics* stat) {
	// standard constants used in MKL library
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;
	clear_local_vector(optimization_data_inst.norms, settings->batch_size); // we use NORMS to store objective values
	// z= B*V
	pXgemm(&transNo, &transNo, &optimization_data_inst.params.DIM_M,
			&settings->batch_size, &optimization_data_inst.params.DIM_N, &one,
			&optimization_data_inst.B[0], &i_one, &i_one,
			optimization_data_inst.descB, optimization_data_inst.V, &i_one,
			&i_one, optimization_data_inst.descV, &zero,
			optimization_data_inst.Z, &i_one, &i_one,
			optimization_data_inst.descZ);
// compute distributed objective values (each computer has only part of the objective value)
	for (int i = 0; i < optimization_data_inst.z_nq; i++) {
		F tmp = 0;
		for (int j = 0; j < optimization_data_inst.z_mp; j++) {

			if (settings->algorithm == solver_structures::L0_constrained_L1_PCA
					|| settings->algorithm
							== solver_structures::L1_constrained_L1_PCA) {
				tmp += abs(
						optimization_data_inst.Z[j
								+ i * optimization_data_inst.z_mp]);
			} else {
				tmp += optimization_data_inst.Z[j
						+ i * optimization_data_inst.z_mp]
						* optimization_data_inst.Z[j
								+ i * optimization_data_inst.z_mp];
			}

		}
		optimization_data_inst.norms[get_column_coordinate(i,
				optimization_data_inst.params.mycol,
				optimization_data_inst.params.npcol,
				optimization_data_inst.params.x_vector_blocking)] = tmp;
	}
	//set Z=sgn(Z)
	if (settings->algorithm == solver_structures::L0_constrained_L1_PCA
			|| settings->algorithm
					== solver_structures::L1_constrained_L1_PCA) {
		vector_sgn(optimization_data_inst.Z, optimization_data_inst.nnz_z);	//y=sgn(y)
	}
	// Multiply V = B'*z
	//		sub(C) := alpha*op(sub(A))*op(sub(B)) + beta*sub(C),
	pXgemm(&trans, &transNo, &optimization_data_inst.params.DIM_N,
			&settings->batch_size, &optimization_data_inst.params.DIM_M, &one,
			&optimization_data_inst.B[0], &i_one, &i_one,
			optimization_data_inst.descB, optimization_data_inst.Z, &i_one,
			&i_one, optimization_data_inst.descZ, &zero,
			optimization_data_inst.V, &i_one, &i_one,
			optimization_data_inst.descV);
	// perform 	threshold operation and compute objective values
	threshold_V_for_constrained(optimization_data_inst, settings, stat);
}

}
}

#endif /* DISTRIBUTED_THRESHOLDINS_H_ */
