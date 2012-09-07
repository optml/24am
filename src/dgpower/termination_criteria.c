/*
 * termination_criteria.h
 *
 *  Created on: Apr 11, 2012
 *      Author: taki
 */

#include "termination_criteria.h"

double computeTheError(const double fval, const double fval_prev) {
	if (fval_prev > fval+1E-3)
		printf("Error dedected: %1.16f < %1.16f\n", fval_prev, fval);
	return (abs(fval - fval_prev) / fval_prev);
}

int termination_criteria(const double error, int it,
		const struct optimization_settings* settings) {
	if (it > 0 && error < settings->toll)
		return 1;
	return 0;
}



