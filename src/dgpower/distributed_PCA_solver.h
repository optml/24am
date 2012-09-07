/*
 * distributed_pca_solver.h
 *
 *  Created on: Mar 30, 2012
 *      Author: taki
 */

#ifndef DISTRIBUTED_PCA_SOLVER_H_
#define DISTRIBUTED_PCA_SOLVER_H_

#include <math.h>
#include <mkl_scalapack.h>
#include "time_helper.h"
#include "mkl_constants_and_headers.h"
#include "optimization_settings.h"
#include "optimization_statistics.h"

#include <stdio.h>
#include <stdlib.h>
#include "termination_criteria.h"
struct value_coordinate_holder {
	double val;
	double prev_val;
};

void distributed_sparse_PCA_solver(double * B, double *x, MDESC descB,
		int nnz_x, MKL_INT ictxt, MKL_INT ROW_BLOCKING,
		MKL_INT X_VECTOR_BLOCKING, struct optimization_settings* settings,
		struct optimization_statistics* stat);

int sgn(double val);

double myabs(double val);

void clear_local_vector(double * v, const int n);

int get_column_coordinate(const int col, const int myCol, const int numCol,
		const int blocking);



int distributed_pca_solver(char* filename, char* outputfile, int MAP_X,
		int input_files, struct optimization_settings* settings,
		struct optimization_statistics* stat);


int distributed_pca_solver_from_two_dim_files(char* filename, char* outputfile,
		int MAP_X, struct optimization_settings* settings,
		struct optimization_statistics* stat);



#endif /* DISTRIBUTED_PCA_SOLVER_H_ */
