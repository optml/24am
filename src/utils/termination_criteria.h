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
 */

/*
 * termination_criteria.h
 *
 *  Created on: Apr 11, 2012
 *      Author: Martin Takac
 */

#ifndef TERMINATION_CRITERIA_H_
#define TERMINATION_CRITERIA_H_

#include "../class/optimization_settings.h"
#include <stdio.h>
#include "various.h"

template<typename F>
F computeTheError(const F fval, const F fval_prev,const solver_structures::optimization_settings* settings) {
#ifdef DEBUG
	if (fval_prev > fval+1E-2 && settings->verbose)
		printf("Error detected: %1.16f < %1.16f\n", fval_prev, fval);
#endif
	return (myabs(fval - fval_prev) / fval_prev);
}

template<typename F>
bool termination_criteria(const F error, int it,
		const solver_structures::optimization_settings* settings) {
	if (it > 0 && error < settings->toll)
		return true;
	return false;
}

template<typename F>
bool termination_criteria(const F fval, const F fval_prev, int it,
		const solver_structures::optimization_settings* settings) {
	if (it > 0 && computeTheError(fval, fval_prev) < settings->toll)
		return true;
	return false;
}

#endif /* TERMINATION_CRITERIA_H_ */
