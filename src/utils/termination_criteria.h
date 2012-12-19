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
F computeTheError(const F fval, const F fval_prev,const SolverStructures::OptimizationSettings* optimizationSettings) {
#ifdef DEBUG
	if (fval_prev > fval+1E-2 && optimizationSettings->verbose)
		printf("Error detected: %1.16f < %1.16f\n", fval_prev, fval);
#endif
	return (myabs(fval - fval_prev) / fval_prev);
}

template<typename F>
bool termination_criteria(const F error, int it,
		const SolverStructures::OptimizationSettings* optimizationSettings) {
	if (it > 0 && error < optimizationSettings->toll)
		return true;
	return false;
}

template<typename F>
bool termination_criteria(const F fval, const F fval_prev, int it,
		const SolverStructures::OptimizationSettings* optimizationSettings) {
	if (it > 0 && computeTheError(fval, fval_prev) < optimizationSettings->toll)
		return true;
	return false;
}

#endif /* TERMINATION_CRITERIA_H_ */
