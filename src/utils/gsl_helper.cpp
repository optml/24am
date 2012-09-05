/*
 *
 *  Created on: Sep 5, 2012
 *      Author: taki
 */

#include "gsl_helper.h"
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

