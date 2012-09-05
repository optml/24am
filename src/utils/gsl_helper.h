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

unsigned int vector_get_nnz(const gsl_vector * x);


#endif /* GLS_HEADERS_H_ */
