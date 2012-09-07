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

#ifndef D_L0_PENALIZED_L2_PCA_H_
#define D_L0_PENALIZED_L2_PCA_H_
#include "mkl_constants_and_headers.h"
#include "optimization_settings.h"
#include "optimization_statistics.h"

void clear_local_vector(double * v, const int n) {
	int i;
	for (i = 0; i < n; i++)
		v[i] = 0;
}

int get_column_coordinate(const int col, const int myCol, const int numCol,
		const int blocking) {
	const int fillup = col / blocking;
	return (fillup * (numCol - 1) + myCol) * blocking + col;
}

void d_l0_penalized_l2_PCA(double * B, double *x, MDESC descB, int nnz_x,
		MKL_INT ictxt, MKL_INT ROW_BLOCKING, MKL_INT X_VECTOR_BLOCKING,
		struct optimization_settings* settings,
		struct optimization_statistics* stat) {
	MKL_INT myrow, mycol, nprow, npcol, info;
	int i, j;
	// create vector "z"
	blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);
	MKL_INT M = descB[2];
	MKL_INT N = descB[3];
	MKL_INT SP_MKL = settings->starting_points;
	MKL_INT z_mp = numroc_(&M, &ROW_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT z_nq = numroc_(&SP_MKL, &ROW_BLOCKING, &mycol, &i_zero, &npcol);
	double * z = (double*) calloc(z_mp * z_nq, sizeof(double));
	for (i = 0; i < z_mp * z_nq; i++) {
		z[i] = -1 + 2 * (double) rand() / RAND_MAX;
	}
	MDESC desc_z;
	MKL_INT i_tmp1 = MAX(1, z_mp);
	descinit_(desc_z, &M, &SP_MKL, &ROW_BLOCKING, &ROW_BLOCKING, &i_zero,
			&i_zero, &ictxt, &i_tmp1, &info);

	//=============== Create description for "V"
	MKL_INT V_mp = numroc_(&N, &X_VECTOR_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT V_nq =
			numroc_(&SP_MKL, &X_VECTOR_BLOCKING, &mycol, &i_zero, &npcol);
	MDESC desc_V;
	i_tmp1 = MAX(1, V_mp);
	descinit_(desc_V, &N, &SP_MKL, &X_VECTOR_BLOCKING, &X_VECTOR_BLOCKING,
			&i_zero, &i_zero, &ictxt, &i_tmp1, &info);
	double * V = (double*) calloc(V_mp * V_nq, sizeof(double));
	//=============== Create description for "x"
	MKL_INT x_mp = numroc_(&N, &X_VECTOR_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT x_nq = numroc_(&i_one, &X_VECTOR_BLOCKING, &mycol, &i_zero, &npcol);
	MDESC desc_x;
	i_tmp1 = MAX(1, x_mp);
	descinit_(desc_x, &N, &i_one, &X_VECTOR_BLOCKING, &X_VECTOR_BLOCKING,
			&i_zero, &i_zero, &ictxt, &i_tmp1, &info);
	//=============== Create description for "norms"
	double * norms =
			(double*) calloc(settings->starting_points, sizeof(double));
	// ======================== RUN SOLVER
	stat->it = 0;
	double fval = 0;
	double fval_prev = 0;
	unsigned int it;
	for (it = 0; it < settings->max_it; it++) {
		stat->it++;
		//================== normalize matriz Z
		clear_local_vector(norms, settings->starting_points);
		//data are stored in column order
		for (i = 0; i < z_nq; i++) {
			double tmp = 0;
			for (j = 0; j < z_mp; j++) {
				tmp += z[j + i * z_mp] * z[j + i * z_mp];
			}
			norms[get_column_coordinate(i, mycol, npcol, X_VECTOR_BLOCKING)]
					= tmp;
		}
		//sum up + distribute norms of "Z"
		dgsum2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER,
				&settings->starting_points, &i_one, norms,
				&settings->starting_points, &i_negone, &i_negone);
		//normalize local "z"
		for (i = 0; i < z_nq; i++) {
			double scaleNorm = 1 / sqrt(
					norms[get_column_coordinate(i, mycol, npcol,
							X_VECTOR_BLOCKING)]);
			for (j = 0; j < z_mp; j++) {
				z[j + i * z_mp] = z[j + i * z_mp] * scaleNorm;
			}
		}
		printf("%d %d :", myrow, mycol);
		for (i = 0; i < settings->starting_points; i++)
			printf("%f ", norms[i]);
		printf("\n");
		//======================
		// Multiply x = B'*z
		//		sub(C) := alpha*op(sub(A))*op(sub(B)) + beta*sub(C),
		pdgemm(&trans, &transNo, &N, &SP_MKL, &M, &one, B, &i_one, &i_one,
				descB, z, &i_one, &i_one, desc_z, &zero, V, &i_one, &i_one,
				desc_V);

		fval = 0;
		for (i = 0; i < nnz_x; i++) {
			double const tmp = x[i];
			double tmp2 = (tmp * tmp - settings->penalty);
			if (tmp2 > 0) {
				fval += tmp2;
			} else {
				x[i] = 0;
			}
		}
		//Agregate FVAL
		dgsum2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER,
				&i_one, &i_one, &fval, &i_one, &i_negone, &i_negone);

		// z= B*V
		pdgemm(&transNo, &transNo, &M, &SP_MKL, &N, &one, B, &i_one, &i_one,
				descB, V, &i_one, &i_one, desc_V, &zero, z, &i_one, &i_one,
				desc_z);
		if (it > 0 && abs((fval - fval_prev) / fval) < settings->toll) { //FIXME CHECK
//			break;
		}
		fval_prev = fval;
	}
	stat->fval = fval;
	//============== COMPUTE final "x" from "z"
	double norm_of_x = 0;
	pdnrm2_(&N, &norm_of_x, x, &i_one, &i_one, &desc_x, &i_one); //FIXME nechapem preco to tu nejde!!!!
	norm_of_x = 1 / norm_of_x;
	for (i = 0; i < nnz_x; i++) {
		x[i] = x[i] * norm_of_x;
	}
	free(z);
	free(V);
	free(norms);
}
#endif /* D_L0_PENALIZED_L2_PCA_H_ */
