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
 *  This file contains a distributed classes to hold data and distributed parameters
 *
 */

#ifndef DISTRIBUTED_CLASSES_H_
#define DISTRIBUTED_CLASSES_H_

#include "mkl_constants_and_headers.h"


namespace SPCASolver {
namespace DistributedClasses {

/*
 * this class is used to hold all parameters for distributed computing
 * and the context for pblas.
 */
class distributed_parameters {
public:
	MKL_INT ictxt; // parallel context
	MKL_INT row_blocking; // determines partition in rows (see PBLAS manual)
	MKL_INT col_blocking; // determines partition in columns (see PBLAS manual)
	MKL_INT x_vector_blocking; // determines partition for vectors (this can have some meaning for
							   // constrained versions. If is set to be "N" one would ged rid of some overhead
	MKL_INT DIM_M; // Total dimension - number of rows of matrix B
	MKL_INT DIM_N; // Total dimension - number of columns of matrix B
	int mycol; // my column position inside of computation grid
	int myrow; // my row position inside of computation grid
	int npcol; // total number of columns in grid
	int nprow; // total number of rows in grid

	// set default parameters. Number 64 is recommended by PBLAS
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

/*
 * This class holds all optimization data needed by the algorithm
 */
template<typename F>
class OptimizationData {
public:
	std::vector<F> x; // final solution will be stored here
	std::vector<F> B; // data for matrix
	F* V; // matrix used to store more starting points (of x-es)
	F* Z; // matrix used to store corresponding vectors "z"
	F* V_constr_threshold; // matrix used to store data to do thresholding (for constrained versions only)
						   // note that only few starting points will be processed on each node
	std::vector<std::vector<F> > V_constr_sort_buffer; // buffer for sorting
	int V_tr_nq; // number of starting points which will be processed on given node (constrained version  only)
	int V_tr_mp; // number of rows == N
	int nnz_v_tr; // total nonzero elements in V_constr_threshold
	bool is_init_data_for_constrained; // local variable to hold info if the data were initialized

	// sizes of matrices V and Z
	int z_nq;
	int z_mp;
	int V_nq;
	int V_mp;
	int nnz_z;
	int nnz_v;

	// Distributed descriptors for matrices
	MDESC descV_threshold;
	MDESC descV;
	MDESC descZ;
	MDESC descB;
	MDESC descx;
	F* norms;
	distributed_parameters params;

	OptimizationData() {
		is_init_data_for_constrained = false;
	}

	// allocate data for constrained versions
	void init_data_for_constrained(
			SolverStructures::OptimizationSettings* settings) {
		if (!this->is_init_data_for_constrained) {
			this->V_tr_mp = numroc_(&this->params.DIM_N, &this->params.DIM_N,
					&this->params.myrow, &i_zero, &this->params.nprow);
			this->V_tr_nq = numroc_(&settings->batch_size, &i_one,
					&this->params.mycol, &i_zero, &this->params.npcol);
			MKL_INT i_tmp1 = MAX(1, this->V_tr_mp);
			MKL_INT info;
			descinit_(this->descV_threshold, &this->params.DIM_N,
					&settings->batch_size, &this->params.DIM_N, &i_one, &i_zero,
					&i_zero, &this->params.ictxt, &i_tmp1, &info);
			this->nnz_v_tr = this->V_tr_mp * this->V_tr_nq;
			this->V_constr_threshold = (F*) calloc(this->nnz_v_tr, sizeof(F));
			if (this->nnz_v_tr > 0) {
				std::vector<F> tmp(this->params.DIM_N);
				V_constr_sort_buffer.resize(this->V_tr_nq, tmp);
			}
			this->is_init_data_for_constrained = true;
		}
	}

	void free_extra_data() {
		if (this->is_init_data_for_constrained) {
			free(this->V_constr_threshold);
			this->is_init_data_for_constrained = false;
		}
	}
};

}
}
#endif
