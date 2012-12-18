/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M. Takac and S. Damla Ahipasaoglu 
 *    "Alternating Maximization: Unified Framework and 24 Parallel Codes for L1 and L2 based Sparse PCA"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 * 
 *  this file contains wrapper for sprace blas. This enables us to use float and double precission
 *
 */

#ifndef MY_SPARSE_WRAPPER_H_
#define MY_SPARSE_WRAPPER_H_

#include "mkl_spblas.h"

char* MY_SPARSE_WRAPPER_TRANS = "T";
char* MY_SPARSE_WRAPPER_NOTRANS = "N";

// matrix matrix multiply
template<typename F>
void my_mm_multiply(bool trans, const int m, const int n, const int experiments,
		const F* vals, const int* row_id, const int* colPtr, const F* means,
		F* x, F*y) {
	for (int ex = 0; ex < experiments; ex++) {
		if (trans) {
			for (int row = 0; row < n; row++)
				y[row + ex * n] = 0;
			//  y=B'*x
			for (int col = 0; col < n; col++) {
				int startMR = row_id[colPtr[col]];
				for (int row = 0; row < startMR; row++) {
					y[col + ex * n] += x[ex * m + row] * means[col];
				}
				int lastMR = 0;
				for (int rowId = colPtr[col]; rowId < colPtr[col + 1];
						rowId++) {
					int tmp_row = row_id[rowId];
					y[col + ex * n] += x[ex * m + tmp_row] * means[col];
					//					y[tmp_row + ex * m] += xval * vals[rowId];
					if (rowId < colPtr[col + 1] - 1) {
						for (int row = tmp_row; row < row_id[rowId + 1];
								row++) {
							//							y[row + ex * m] += xval * means[col];
							y[col + ex * n] += x[ex * m + row] * means[col];
						}
					}
					lastMR = row_id[rowId];
				}
				for (int row = lastMR + 1; row < m; row++) {
					y[col + ex * n] += x[ex * m + row] * means[col];
				}

			}

		} else {
			for (int row = 0; row < m; row++)
				y[row + ex * m] = 0;
			//  y=B*x
			for (int col = 0; col < n; col++) {
				F xval = x[col + ex * n];
				if (xval != 0) {
					int startMR = row_id[colPtr[col]];
					for (int row = 0; row < startMR; row++) {
						y[row + ex * m] += xval * means[col];
					}
					int lastMR = 0;
					for (int rowId = colPtr[col]; rowId < colPtr[col + 1];
							rowId++) {
						int tmp_row = row_id[rowId];
						y[tmp_row + ex * m] += xval * vals[rowId];
						if (rowId < colPtr[col + 1] - 1) {
							for (int row = tmp_row; row < row_id[rowId + 1];
									row++) {
								y[row + ex * m] += xval * means[col];
							}
						}
						lastMR = row_id[rowId];
					}
					for (int row = lastMR + 1; row < m; row++) {
						y[row + ex * m] += xval * means[col];
					}

				}
			}
		}
	}
}

void sparse_matrix_matrix_multiply(char *transa, int mI, int nI, int kI,
		double *alpha, char *matdescra, double *val, MKL_INT *indx,
		MKL_INT *pntrb, MKL_INT *pntre, double *b, int ldbI, double *beta,
		double *c, int ldcI) {

	MKL_INT m = mI;
	MKL_INT n = nI;
	MKL_INT k = kI;
	MKL_INT ldc = ldcI;
	MKL_INT ldb = ldbI;

	mkl_dcscmm(transa, &m, &n, &k, alpha, matdescra, val, indx, pntrb, pntre, b,
			&ldb, beta, c, &ldc);
}

void sparse_matrix_matrix_multiply(char *transa, int mI, int nI, int kI,
		float *alpha, char *matdescra, float *val, MKL_INT *indx,
		MKL_INT *pntrb, MKL_INT *pntre, float *b, int ldbI, float *beta,
		float *c, int ldcI) {

	MKL_INT m = mI;
	MKL_INT n = nI;
	MKL_INT k = kI;
	MKL_INT ldc = ldcI;
	MKL_INT ldb = ldbI;

	mkl_scscmm(transa, &m, &n, &k, alpha, matdescra, val, indx, pntrb, pntre, b,
			&ldb, beta, c, &ldc);
}
#endif /* MY_CBLAS_WRAPPER_H_ */
