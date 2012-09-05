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
void cblas_vector_scale(const int n, F* vector, const F factor);

template<>
void cblas_vector_scale(const int n, double* vector, const double factor);

template<>
void cblas_vector_scale(const int n, float* vector, const float factor);

void cblas_matrix_matrix_multiply(const CBLAS_ORDER Order,
		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
		const int N, const int K, const double alpha, const double *A,
		const int lda, const double *B, const int ldb, const double beta,
		double *C, const int ldc);

void cblas_matrix_matrix_multiply(const CBLAS_ORDER Order,
		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
		const int N, const int K, const float alpha, const float *A,
		const int lda, const float *B, const int ldb, const float beta,
		float *C, const int ldc);

double cblas_l1_norm(const int N, const double *X, const int incX);

float cblas_l1_norm(const int N, const float *X, const int incX);

double cblas_l2_norm(const int N, const double *X, const int incX);
float cblas_l2_norm(const int N, const float *X, const int incX);

void cblas_vector_copy(const int N, const double *X, const int incX, double *Y,
		const int incY);

void cblas_vector_copy(const int N, const float *X, const int incX, float *Y,
		const int incY);

CBLAS_INDEX cblas_vector_max_index(const int N, const double *X,
		const int incX);

CBLAS_INDEX cblas_vector_max_index(const int N, const float *X, const int incX);




#endif /* MY_CBLAS_WRAPPER_H_ */
