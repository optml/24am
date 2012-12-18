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
 */

/*
 * my_cblas_wrapper.h
 *
 *  Created on: Apr 14, 2012
 *      Author: taki
 */

#ifndef MY_CBLAS_WRAPPER_H_
#define MY_CBLAS_WRAPPER_H_

#include <gsl/gsl_cblas.h>


template<typename F>
void cblas_vector_scale(const int n, F* vector, const F factor) {
}

template<>
void cblas_vector_scale(const int n, double* vector, const double factor) {
	cblas_dscal(n, factor, vector, 1);
}

template<>
void cblas_vector_scale(const int n, float* vector, const float factor) {
	cblas_sscal(n, factor, vector, 1);
}

//template<typename F>
//void cblas_matrix_matrix_multiply(const CBLAS_ORDER Order,
//		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
//		const int M, const int N, const int K, const F alpha,
//		const F *A, const int lda, const F *B, const int ldb,
//		const F beta, F *C, const int ldc) {
//}

//template<>
void cblas_matrix_matrix_multiply(const CBLAS_ORDER Order,
		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
		const int M, const int N, const int K, const double alpha,
		const double *A, const int lda, const double *B, const int ldb,
		const double beta, double *C, const int ldc) {
	cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
			ldc);
}

//template<>
void cblas_matrix_matrix_multiply(const CBLAS_ORDER Order,
		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
		const int M, const int N, const int K, const float alpha,
		const float *A, const int lda, const float *B, const int ldb,
		const float beta, float *C, const int ldc) {
	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
			ldc);
}

double cblas_l1_norm(const int N, const double *X, const int incX) {
	return cblas_dasum(N, X, incX);
}

float cblas_l1_norm(const int N, const float *X, const int incX) {
	return cblas_sasum(N, X, incX);
}

double cblas_l2_norm(const int N, const double *X, const int incX) {
	return cblas_dnrm2(N, X, incX);
}

float cblas_l2_norm(const int N, const float *X, const int incX) {
	return cblas_snrm2(N, X, incX);
}

void cblas_vector_copy(const int N, const double *X, const int incX,
		double *Y, const int incY) {
	cblas_dcopy(N, X, incX, Y, incY);
}

void cblas_vector_copy(const int N, const float *X, const int incX,
		float *Y, const int incY) {
	cblas_scopy(N, X, incX, Y, incY);
}



CBLAS_INDEX cblas_vector_max_index(const int N, const double *X, const int incX){
	return cblas_idamax(N, X, incX);
}

CBLAS_INDEX cblas_vector_max_index(const int N, const float *X, const int incX){
	return cblas_isamax(N, X, incX);
}




#endif /* MY_CBLAS_WRAPPER_H_ */
