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
 */
/*
 * gls_headers.h
 *
 *  Created on: Sep 5, 2012
 *      Author: taki
 */

#ifndef GLS_HEADERS_H_
#define GLS_HEADERS_H_


#define GSL_RANGE_CHECK 0

#include <gsl/gsl_cblas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

unsigned int vector_get_nnz(const gsl_vector * x) {
	unsigned int nnz = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:nnz)
#endif
	for (unsigned int i = 0; i < x->size; i++) {
		if (gsl_vector_get(x, i) != 0)
			nnz++;
	}
	return nnz;
}


#endif /* GLS_HEADERS_H_ */
