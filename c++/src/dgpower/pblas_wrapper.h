/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine
 * and is based on the paper
 *    P. Richtarik, M.Jahani, S. Damla Ahipasaoglu and M. Takac
 *    "Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 *
 *  Created on: Sep 21, 2012
 *      Author: taki
 *
 *
 *
 *  This file contains a wrappers for Parallel BLAS. This allows us to have one code
 *  for double and also for floats.
 *
 */


#ifndef PBLAS_WAPPER_H_
#define PBLAS_WAPPER_H_

void pXgeadd(char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a,
		MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c,
		MKL_INT *ic, MKL_INT *jc, MKL_INT *descc) {
	psgeadd_(trans, m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void pXgeadd(char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a,
		MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c,
		MKL_INT *ic, MKL_INT *jc, MKL_INT *descc) {
	pdgeadd_(trans, m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void pXgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k,
		float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca,
		float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta,
		float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc) {
	psgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb,
			beta, c, ic, jc, descc);

}
void pXgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k,
		double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca,
		double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta,
		double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc) {
	pdgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb,
			beta, c, ic, jc, descc);
}

void pXnrm2(MKL_INT *n, float *norm2, float *x, MKL_INT *ix, MKL_INT *jx,
		MKL_INT *descx, MKL_INT *incx) {
	psnrm2_(n, norm2, x, ix, jx, descx, incx);
}
void pXnrm2(MKL_INT *n, double *norm2, double *x, MKL_INT *ix, MKL_INT *jx,
		MKL_INT *descx, MKL_INT *incx) {
	pdnrm2_(n, norm2, x, ix, jx, descx, incx);
}

void Xgsum2d(MKL_INT *ConTxt, char *scope, char *top, MKL_INT *m, MKL_INT *n,
		float *A, MKL_INT *lda, MKL_INT *rdest, MKL_INT *cdest) {
	sgsum2d_(ConTxt, scope, top, m, n, A, lda, rdest, cdest);
}

void Xgsum2d(MKL_INT *ConTxt, char *scope, char *top, MKL_INT *m, MKL_INT *n,
		double *A, MKL_INT *lda, MKL_INT *rdest, MKL_INT *cdest) {
	dgsum2d_(ConTxt, scope, top, m, n, A, lda, rdest, cdest);
}

#endif /* PBLAS_WAPPER_H_ */
