/*
//HEADER INFO
 */

#ifndef MY_SPARSE_WRAPPER_H_
#define MY_SPARSE_WRAPPER_H_

#include "mkl_spblas.h"

char* MY_SPARSE_WRAPPER_TRANS = "T";
char* MY_SPARSE_WRAPPER_NOTRANS = "N";

template<typename F>
void my_mm_multiply(bool trans, const int m, const int n,
		const int experiments, const F* vals, const int* row_id,
		const int* colPtr, const F* means, F* x, F*y) {
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
				for (int rowId = colPtr[col]; rowId < colPtr[col + 1]; rowId++) {
					int tmp_row = row_id[rowId];
					y[col + ex * n] += x[ex * m + tmp_row] * means[col];
					//					y[tmp_row + ex * m] += xval * vals[rowId];
					if (rowId < colPtr[col + 1] - 1) {
						for (int row = tmp_row; row < row_id[rowId + 1]; row++) {
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
					for (int rowId = colPtr[col]; rowId < colPtr[col + 1]; rowId++) {
						int tmp_row = row_id[rowId];
						y[tmp_row + ex * m] += xval * vals[rowId];
						if (rowId < colPtr[col + 1] - 1) {
							for (int row = tmp_row; row < row_id[rowId + 1]; row++) {
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

//template<>
void sparse_matrix_matrix_multiply(char *transa, int mI, int nI, int kI,
		double *alpha, char *matdescra, double *val, MKL_INT *indx,
		MKL_INT *pntrb, MKL_INT *pntre, double *b, int ldbI, double *beta,
		double *c, int ldcI) {

	MKL_INT m = mI;
	MKL_INT n = nI;
	MKL_INT k = kI;
	MKL_INT ldc = ldcI;
	MKL_INT ldb = ldbI;

	//	for (int i = 0; i < 5; i++) {
	//		printf("%d PTR %d  %d==%d \n",i, pntrb[i],pntrb[i+1],pntre[i]);
	//		for (int row = pntrb[i]; row < pntrb[i+1]; row++) {
	//			printf("%d  %d   %f\n", indx[row], i, val[row]);
	//		}
	//	}

	mkl_dcscmm(transa, &m, &n, &k, alpha, matdescra, val, indx, pntrb, pntre,
			b, &ldb, beta, c, &ldc);
}

//void sparse_matrix_matrix_multiply(char *transa, MKL_INT *m, MKL_INT *n,
//		float *alpha, char *matdescra, float *val, MKL_INT *indx,
//		MKL_INT *pntrb, MKL_INT *pntre, float *b, MKL_INT *ldb, float *c,
//		MKL_INT *ldc) {
//	mkl_scscmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b,
//			ldb, beta, c, ldc);
//}

//template<>
//void sparse_matrix_matrix_multiply(const CBLAS_ORDER Order,
//		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
//		const int M, const int N, const int K, const float alpha,
//		const float *A, const int lda, const float *B, const int ldb,
//		const float beta, float *C, const int ldc) {
//	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
//			ldc);
//}


#endif /* MY_CBLAS_WRAPPER_H_ */
