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
 *  This file contains a distributed solver. The solver works only for dense matrices
 *  and doesn't have implemented on the fly strategy and batching with more than one batch.
 *
 */

#ifndef DISTRIBUTED_PCA_SOLVER_H_
#define DISTRIBUTED_PCA_SOLVER_H_

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
#include "distributed_thresholdings.h"

namespace SPCASolver {
namespace DistributedSolver {

template<typename F>
void denseDataSolver(
		SPCASolver::DistributedClasses::OptimizationData<F>& optimization_data_inst,
		SolverStructures::OptimizationSettings* settings,
		SolverStructures::OptimizationStatistics* stat) {
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;
	MKL_INT myrow, mycol, nprow, npcol, info;
	MKL_INT ictxt = optimization_data_inst.params.ictxt;
	MKL_INT M = optimization_data_inst.params.DIM_M;
	MKL_INT N = optimization_data_inst.params.DIM_N;
	blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);

	if (settings->verbose && settings->proccess_node == 0) {
		std::cout << "Solver started " << std::endl;
	}
	settings->chceckInputAndModifyIt(N);
	stat->it = settings->max_it;
	// Allocate vector for stat to return which point needs how much iterations
	if (settings->get_it_for_all_points) {
		stat->iters.resize(settings->starting_points, -1);
		stat->cardinalities.resize(settings->starting_points, -1);
		stat->values.resize(settings->starting_points, -1);

	}
	const unsigned int number_of_experiments_per_batch = settings->batch_size;
	// TODO implement on the fly and other strategies...

	F* B = &optimization_data_inst.B[0];
	F* x = &optimization_data_inst.x[0];

	MKL_INT ROW_BLOCKING = optimization_data_inst.params.row_blocking;
	MKL_INT X_VECTOR_BLOCKING = optimization_data_inst.params.x_vector_blocking;

	int i, j;
	// create vector "z"
	optimization_data_inst.z_mp = numroc_(&M, &ROW_BLOCKING, &myrow, &i_zero,
			&nprow);
	optimization_data_inst.z_nq = numroc_(&settings->batch_size, &ROW_BLOCKING,
			&mycol, &i_zero, &npcol);

	unsigned int seed = mycol * nprow + myrow;
	optimization_data_inst.nnz_z = optimization_data_inst.z_mp
			* optimization_data_inst.z_nq;
	optimization_data_inst.Z = (F*) calloc(optimization_data_inst.nnz_z,
			sizeof(F));
	MKL_INT i_tmp1 = MAX(1, optimization_data_inst.z_mp);
	descinit_(optimization_data_inst.descZ, &M, &settings->batch_size,
			&ROW_BLOCKING, &ROW_BLOCKING, &i_zero, &i_zero, &ictxt, &i_tmp1,
			&info);
	//=============== Create description for "V"
	optimization_data_inst.V_mp = numroc_(&N, &X_VECTOR_BLOCKING, &myrow,
			&i_zero, &nprow);
	optimization_data_inst.V_nq = numroc_(&settings->batch_size,
			&X_VECTOR_BLOCKING, &mycol, &i_zero, &npcol);
	i_tmp1 = MAX(1, optimization_data_inst.V_mp);
	descinit_(optimization_data_inst.descV, &N, &settings->batch_size,
			&X_VECTOR_BLOCKING, &X_VECTOR_BLOCKING, &i_zero, &i_zero, &ictxt,
			&i_tmp1, &info);
	optimization_data_inst.nnz_v = optimization_data_inst.V_mp
			* optimization_data_inst.V_nq;
	optimization_data_inst.V = (F*) calloc(optimization_data_inst.nnz_v,
			sizeof(F));
	for (i = 0; i < optimization_data_inst.nnz_v; i++) {
		optimization_data_inst.V[i] = -1 + 2 * (F) rand_r(&seed) / RAND_MAX;
	}
	// initial thresholding of matrix "V".....
	if (settings->isConstrainedProblem()) {
		SPCASolver::distributed_thresholdings::threshold_V_for_constrained(optimization_data_inst, settings, stat);
	}

	//=============== Create description for "x"
	MKL_INT x_mp = numroc_(&N, &X_VECTOR_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT x_nq = numroc_(&i_one, &X_VECTOR_BLOCKING, &mycol, &i_zero, &npcol);
	MDESC desc_x;

	i_tmp1 = MAX(1, x_mp);
	descinit_(desc_x, &N, &i_one, &X_VECTOR_BLOCKING, &X_VECTOR_BLOCKING,
			&i_zero, &i_zero, &ictxt, &i_tmp1, &info);
	//=============== Create vector for "norms"
	optimization_data_inst.norms = (F*) calloc(settings->starting_points,
			sizeof(F));
	std::vector<value_coordinate_holder<F> > values(settings->starting_points);
	// ======================== RUN SOLVER
	stat->it = 0;
	double fval = 0;
	double fval_prev = 0;
	unsigned int it;
	for (it = 0; it < settings->max_it; it++) {
		stat->it++;
		if (settings->isConstrainedProblem()) {
			SPCASolver::distributed_thresholdings::perform_one_distributed_iteration_for_constrained_pca(
					optimization_data_inst, settings, stat);
		} else {
			SPCASolver::distributed_thresholdings::perform_one_distributed_iteration_for_penalized_pca(
					optimization_data_inst, settings, stat);
		}
		//Agregate FVAL
		Xgsum2d(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER,
				&settings->starting_points, &i_one,
				optimization_data_inst.norms, &settings->starting_points,
				&i_negone, &i_negone);
		double max_error = 0;
		for (i = 0; i < settings->starting_points; i++) {
			if (settings->algorithm == SolverStructures::L0_penalized_L1_PCA
					|| settings->algorithm
							== SolverStructures::L0_penalized_L2_PCA
					|| settings->algorithm
							== SolverStructures::L0_constrained_L1_PCA
					|| settings->algorithm
							== SolverStructures::L1_constrained_L1_PCA) {
				values[i].val = optimization_data_inst.norms[i];
			} else {
				values[i].val = sqrt(optimization_data_inst.norms[i]);
			}
			if (it > 0) {
				double tmp_error = computeTheError(values[i].val,
						values[i].prev_val, settings);
				if (tmp_error > max_error)
					max_error = tmp_error;
			}
			values[i].prev_val = values[i].val;
		}

		if (it > 0 && termination_criteria(max_error, it, settings)) { //FIXME CHECK
			break;
		}
	}

	stat->fval = -1;
	int max_selection_idx = -1;
	for (i = 0; i < settings->starting_points; i++) {
		if (values[i].val > stat->fval) {
			max_selection_idx = i;
			stat->fval = values[i].val;
		}
	}
	//copy "MAX_SELECTION_IDX" column from matrix V into vector x!
	//	sub(C):=beta*sub(C) + alpha*op(sub(A)),
	max_selection_idx++;	// because next fucntion use 1-based
	pXgeadd(&transNo, &N, &i_one, &one, optimization_data_inst.V, &i_one,
			&max_selection_idx, optimization_data_inst.descV, &zero, x, &i_one,
			&i_one, desc_x);
	//============== COMPUTE final "x"
	F norm_of_x = 0;
	pXnrm2(&N, &norm_of_x, x, &i_one, &i_one, desc_x, &i_one);
	norm_of_x = 1 / norm_of_x;
	for (i = 0; i < optimization_data_inst.x.size(); i++) {
		x[i] = x[i] * norm_of_x;
	}
	optimization_data_inst.free_extra_data();
	free(optimization_data_inst.Z);
	free(optimization_data_inst.V);
	free(optimization_data_inst.norms);
}

template<typename F>
int loadDataFrom2DFilesAndDistribute(
		SPCASolver::DistributedClasses::OptimizationData<F> &optimization_data_inst,
		SolverStructures::OptimizationSettings* settings,
		SolverStructures::OptimizationStatistics* stat) {
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;
	MKL_INT X_VECTOR_BLOCKING = optimization_data_inst.params.x_vector_blocking;
	MKL_INT ROW_BLOCKING = optimization_data_inst.params.row_blocking;
	MKL_INT COL_BLOCKING = optimization_data_inst.params.col_blocking;
	char* filename = settings->data_file;
	char* outputfile = settings->result_file;
	MKL_INT iam, nprocs, ictxt, ictxt2, myrow, mycol, nprow, npcol;
	MKL_INT info;
	MKL_INT m, n, nb, mb, mp, nq, lld, lld_local;
	int i, j, k;
	blacs_pinfo_(&iam, &nprocs);
	blacs_get_(&i_negone, &i_zero, &ictxt);
	int MAP_X = settings->distributed_row_grid_file;
	int MAP_Y = nprocs / MAP_X;
	if (MAP_X * MAP_Y != nprocs) {
		if (iam == 0)
			printf("Wrong Grid Map specification!  %d %d\n", MAP_X, MAP_Y);
		return -1;
	}
	blacs_gridinit_(&ictxt, "C", &MAP_X, &MAP_Y); // Create row map
	blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);
	optimization_data_inst.params.mycol = mycol;
	optimization_data_inst.params.npcol = npcol;
	optimization_data_inst.params.myrow = myrow;
	optimization_data_inst.params.nprow = nprow;

	optimization_data_inst.params.ictxt = ictxt;

	/* ===========================================================================================
	 *                LOAD DATA FROM FILES AND DISTRIBUTE IT ACROS NODES
	 *
	 * ===========================================================================================
	 */
	// Load data from files
	char final_file[1000];
	sprintf(final_file, "%s%d-%d", filename, myrow, mycol);
	int DIM_M = 0;
	int DIM_N = 0;
	int DIM_N_LOCAL = 0;
	int DIM_M_LOCAL = 0;
	F * B_Local;
	blacs_barrier_(&ictxt, &C_CHAR_SCOPE_ALL);
	double start_time = gettime();

	FILE * fin = fopen(final_file, "r");
	if (fin == NULL) {
		B_Local = (F*) calloc(0, sizeof(F));
	} else {
		fscanf(fin, "%d;%d", &DIM_M_LOCAL, &DIM_N_LOCAL);
		B_Local = (F*) calloc(DIM_M_LOCAL * DIM_N_LOCAL, sizeof(F));
		for (j = 0; j < DIM_M_LOCAL; j++) {
			for (i = 0; i < DIM_N_LOCAL; i++) {
				float tmp = -1;
				fscanf(fin, "%f;", &tmp);
				B_Local[i * DIM_M_LOCAL + j] = tmp;
			}
		}
		fclose(fin);
	}
	double end_time = gettime();
	if (iam == 0) {
		printf("loading data from file into memmory took %f\n",
				end_time - start_time);
	}

	//-------------------------------
	blacs_barrier_(&ictxt, &C_CHAR_SCOPE_ALL);
	start_time = gettime();

	i_tmp1 = -1, i_tmp2 = -1;
	DIM_N = DIM_N_LOCAL;
	int DIM_N_INPUT_BLOCKING = DIM_N_LOCAL;
	igsum2d_(&ictxt, &C_CHAR_SCOPE_ROWS, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_N, &i_one, &i_negone, &i_negone);

	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_N, &i_one, &i_tmp1, &i_tmp2, &i_one, &i_negone,
			&i_negone);

	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_N_INPUT_BLOCKING, &i_one, &i_tmp1, &i_tmp2, &i_one,
			&i_negone, &i_negone);

	DIM_M = DIM_M_LOCAL;
	igsum2d_(&ictxt, &C_CHAR_SCOPE_COLS, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_M, &i_one, &i_negone, &i_negone);
	MKL_INT DIM_MM = DIM_M;
	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_MM, &i_one, &i_tmp1, &i_tmp2, &i_one, &i_negone,
			&i_negone);
	DIM_M = DIM_MM;

	MKL_INT B_Local_row_blocking = DIM_M_LOCAL;
	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &B_Local_row_blocking, &i_one, &i_tmp1, &i_tmp2, &i_one,
			&i_negone, &i_negone);

	// Now, the size of Matrix B is   DIM_M x DIM_n
	/*  Matrix descriptors */
	MDESC descB_local;
	/* Create Local Descriptors + Global Descriptors*/
	i_tmp1 = numroc_(&DIM_M, &B_Local_row_blocking, &myrow, &i_zero, &nprow);
	i_tmp1 = MAX(1, i_tmp1);
	descinit_(descB_local, &DIM_M, &DIM_N, &B_Local_row_blocking,
			&DIM_N_INPUT_BLOCKING, &i_zero, &i_zero, &ictxt, &i_tmp1, &info);

	mp = numroc_(&DIM_M, &ROW_BLOCKING, &myrow, &i_zero, &nprow);
	nq = numroc_(&DIM_N, &COL_BLOCKING, &mycol, &i_zero, &npcol);
	lld = MAX(mp, 1);

	descinit_(optimization_data_inst.descB, &DIM_M, &DIM_N, &ROW_BLOCKING,
			&COL_BLOCKING, &i_zero, &i_zero, &ictxt, &lld, &info);

	end_time = gettime();
	if (iam == 0) {
		printf("allocate descriptors and vectors %f\n", end_time - start_time);
	}

	// Distribute data from BLocal => B
	blacs_barrier_(&ictxt, &C_CHAR_SCOPE_ALL);
	start_time = gettime();

	optimization_data_inst.B.resize(mp * nq);
	pXgeadd(&transNo, &DIM_M, &DIM_N, &one, B_Local, &i_one, &i_one,
			descB_local, &zero, &optimization_data_inst.B[0], &i_one, &i_one,
			optimization_data_inst.descB);
	free(B_Local);

	end_time = gettime();
	if (iam == 0) {
		printf("matrix distribution accross the grid took %f\n",
				end_time - start_time);
	}

	/* =============================================================
	 *         Initialize vector "x" where solution will be stored
	 *
	 * =============================================================
	 */

	MKL_INT x_mp = numroc_(&DIM_N, &X_VECTOR_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT x_nq = numroc_(&i_one, &X_VECTOR_BLOCKING, &mycol, &i_zero, &npcol);
	optimization_data_inst.x.resize(x_mp * x_nq);

	i_tmp1 = MAX(1, x_mp);
	descinit_(optimization_data_inst.descx, &DIM_N, &i_one, &X_VECTOR_BLOCKING,
			&X_VECTOR_BLOCKING, &i_zero, &i_zero, &ictxt, &i_tmp1, &info);

	optimization_data_inst.params.DIM_M = DIM_M;
	optimization_data_inst.params.DIM_N = DIM_N;

	return 0;
}

template<typename F>
int gather_and_store_best_result_to_file(
		SPCASolver::DistributedClasses::OptimizationData<F> &optimization_data_inst,
		SolverStructures::OptimizationSettings* settings,
		SolverStructures::OptimizationStatistics* stat) {
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;
	/* =============================================================
	 *          STORE RESULT
	 * =============================================================
	 */
	double start_time = gettime();
	MKL_INT iam, nprocs;
	blacs_pinfo_(&iam, &nprocs);
	MKL_INT myrow, mycol, nprow, npcol, info;
	MKL_INT ictxt = optimization_data_inst.params.ictxt;
	blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);

	MKL_INT DIM_N = optimization_data_inst.params.DIM_N;
	F* x_local = NULL;
	if (iam == 0) {
		x_local = (F*) calloc(DIM_N, sizeof(F));
	} else {
		x_local = (F*) calloc(0, sizeof(F));
	}
	MDESC desc_x_local;
	MKL_INT x_local_mq = numroc_(&DIM_N, &DIM_N, &myrow, &i_zero, &nprow);
	MKL_INT x_local_np = numroc_(&i_one, &i_one, &mycol, &i_zero, &npcol);
	i_tmp1 = MAX(1, x_local_mq);
	descinit_(desc_x_local, &DIM_N, &i_one, &DIM_N, &i_one, &i_zero, &i_zero,
			&ictxt, &i_tmp1, &info);
	pXgeadd(&transNo, &DIM_N, &i_one, &one, &optimization_data_inst.x[0],
			&i_one, &i_one, optimization_data_inst.descx, &zero, x_local,
			&i_one, &i_one, desc_x_local);
	if (iam == 0) {
		FILE * fin = fopen(settings->result_file, "w");
		for (int i = 0; i < DIM_N; i++) {
			fprintf(fin, "%f;", x_local[i]);
		}
		fclose(fin);
	}

	double end_time = gettime();
	if (iam == 0) {
		printf("storing result %f\n", end_time - start_time);
	}

	free(x_local);
	return iam;
}
}
}

#endif /* DISTRIBUTED_PCA_SOLVER_H_ */
