/*
 //HEADER INFO
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
#include "../utils/tresh_functions.h"

template<typename F>
void clear_local_vector(F * v, const int n) {
	int i;
	for (i = 0; i < n; i++)
		v[i] = 0;
}

int get_column_coordinate(const int col, const int myCol, const int numCol,
		const int blocking) {
	const int fillup = col / blocking;
	return (fillup * (numCol - 1) + myCol) * blocking + col;
}

namespace PCA_solver {
namespace distributed_solver {

class distributed_parameters {
public:
	MKL_INT ictxt;
	MKL_INT row_blocking;
	MKL_INT col_blocking;
	MKL_INT x_vector_blocking;
	MKL_INT DIM_M;
	MKL_INT DIM_N;
	int mycol;
	int myrow;
	int npcol;
	int nprow;

	distributed_parameters() {
		row_blocking = 64;
		col_blocking = 64;
		x_vector_blocking = 64;
		mycol = 0;
		npcol = 0;
		myrow = 0;
		nprow = 0;
	}
};

template<typename F>
class optimization_data {
public:
	std::vector<F> x;
	std::vector<F> B;
	F* V;
	F* Z;
	F* V_constr_treshold;
	std::vector<std::vector<F> > V_constr_sort_buffer;
	int V_tr_nq;
	int V_tr_mp;
	bool is_init_data_for_constrained;
	int nnz_v_tr;

	int nnz_z;
	int nnz_v;

	int z_nq;
	int z_mp;

	int V_nq;
	int V_mp;
	MDESC descV_treshold;
	MDESC descV;
	MDESC descZ;
	MDESC descB;
	MDESC descx;
	F* norms;
	distributed_parameters params;

	optimization_data() {
		is_init_data_for_constrained = false;
	}

	void init_data_for_constrained(
			solver_structures::optimization_settings* settings) {
		if (!this->is_init_data_for_constrained) {

			this->V_tr_mp = numroc_(&this->params.DIM_N, &this->params.DIM_N,
					&this->params.myrow, &i_zero, &this->params.nprow);
			this->V_tr_nq = numroc_(&settings->batch_size, &i_one,
					&this->params.mycol, &i_zero, &this->params.npcol);

			MKL_INT i_tmp1 = MAX(1, this->V_tr_mp);
			MKL_INT info;
			descinit_(this->descV_treshold, &this->params.DIM_N,
					&settings->batch_size, &this->params.DIM_N, &i_one, &i_zero,
					&i_zero, &this->params.ictxt, &i_tmp1, &info);
			this->nnz_v_tr = this->V_tr_mp * this->V_tr_nq;
			this->V_constr_treshold = (F*) calloc(this->nnz_v_tr, sizeof(F));
			if (this->nnz_v_tr > 0) {
				std::vector<F> tmp(this->params.DIM_N);
				V_constr_sort_buffer.resize(this->V_tr_nq, tmp);
			}
			this->is_init_data_for_constrained = true;
		}
	}

	void free_extra_data() {
		if (this->is_init_data_for_constrained) {
			free(this->V_constr_treshold);
			this->is_init_data_for_constrained=false;
		}

	}

};

template<typename F>
void perform_one_distributed_iteration_for_penalized_pca(
		PCA_solver::distributed_solver::optimization_data<F>& optimization_data_inst,
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
//		sum up + distribute norms of "Z"
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
template<typename F>
void threshold_V_for_constrained(
		PCA_solver::distributed_solver::optimization_data<F>& optimization_data_inst,
		solver_structures::optimization_settings* settings,
		solver_structures::optimization_statistics* stat) {
	F zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0e+0;
	//================== Treshhold matrix V
	optimization_data_inst.init_data_for_constrained(settings);
	// obtain V from all cluster into V_constr_treshold for sorting and tresholding
	pXgeadd(&transNo, &optimization_data_inst.params.DIM_N,
			&settings->batch_size, &one, optimization_data_inst.V, &i_one,
			&i_one, optimization_data_inst.descV, &zero,
			optimization_data_inst.V_constr_treshold, &i_one, &i_one,
			optimization_data_inst.descV_treshold);
	//compute tresholding
	if (optimization_data_inst.V_tr_mp == optimization_data_inst.params.DIM_N) {
		for (unsigned int j = 0; j < optimization_data_inst.V_tr_nq; j++) {
			F norm_of_x;
			if (settings->isL1ConstrainedProblem()) {
				norm_of_x =
						soft_tresholding(
								&optimization_data_inst.V_constr_treshold[optimization_data_inst.params.DIM_N
										* j],
								optimization_data_inst.params.DIM_N,
								settings->constrain,
								optimization_data_inst.V_constr_sort_buffer[j],settings); // x = S_w(x)
			} else {
				settings->hard_tresholding_using_sort = true;
				norm_of_x =
						k_hard_tresholding(
								&optimization_data_inst.V_constr_treshold[optimization_data_inst.params.DIM_N
										* j],
								optimization_data_inst.params.DIM_N,
								settings->constrain,
								optimization_data_inst.V_constr_sort_buffer[j],
								settings); // x = T_k(x)
			}

			if (settings->algorithm == solver_structures::L0_constrained_L2_PCA
					|| settings->algorithm
							== solver_structures::L1_constrained_L2_PCA) {
				cblas_vector_scale(optimization_data_inst.params.DIM_N,
						&optimization_data_inst.V_constr_treshold[optimization_data_inst.params.DIM_N
								* j], 1 / norm_of_x);
			}
		}
	}
	//return thresholded values
	pXgeadd(&transNo, &optimization_data_inst.params.DIM_N,
			&settings->batch_size, &one,
			optimization_data_inst.V_constr_treshold, &i_one, &i_one,
			optimization_data_inst.descV_treshold, &zero,
			optimization_data_inst.V, &i_one, &i_one,
			optimization_data_inst.descV);
}

template<typename F>
void perform_one_distributed_iteration_for_constrained_pca(
		PCA_solver::distributed_solver::optimization_data<F>& optimization_data_inst,
		solver_structures::optimization_settings* settings,
		solver_structures::optimization_statistics* stat) {
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
	//set Z=sgn(Z)
	if (settings->algorithm == solver_structures::L0_constrained_L1_PCA
			|| settings->algorithm
					== solver_structures::L1_constrained_L1_PCA) {
		vector_sgn(optimization_data_inst.Z, optimization_data_inst.nnz_z);	//y=sgn(y)
	}

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

template<typename F>
void distributed_sparse_PCA_solver(
		PCA_solver::distributed_solver::optimization_data<F>& optimization_data_inst,
		solver_structures::optimization_settings* settings,
		solver_structures::optimization_statistics* stat) {
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
	// initial tresholding of matrix "V".....
	if (settings->isConstrainedProblem()) {
		threshold_V_for_constrained(optimization_data_inst, settings, stat);
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
			perform_one_distributed_iteration_for_constrained_pca(
					optimization_data_inst, settings, stat);
		} else {
			perform_one_distributed_iteration_for_penalized_pca(
					optimization_data_inst, settings, stat);
		}
		//Agregate FVAL
		Xgsum2d(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER,
				&settings->starting_points, &i_one,
				optimization_data_inst.norms, &settings->starting_points,
				&i_negone, &i_negone);
		double max_error = 0;
		for (i = 0; i < settings->starting_points; i++) {
			if (settings->algorithm == solver_structures::L0_penalized_L1_PCA
					|| settings->algorithm
							== solver_structures::L0_penalized_L2_PCA) {
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
int load_data_from_2d_files_and_distribution(
		PCA_solver::distributed_solver::optimization_data<F> &optimization_data_inst,
		solver_structures::optimization_settings* settings,
		solver_structures::optimization_statistics* stat) {
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
		PCA_solver::distributed_solver::optimization_data<F> &optimization_data_inst,
		solver_structures::optimization_settings* settings,
		solver_structures::optimization_statistics* stat) {
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
