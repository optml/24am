/*
 * termination_criteria.h
 *
 *  Created on: Apr 11, 2012
 *      Author: taki
 */

#ifndef TERMINATION_CRITERIA_H_
#define TERMINATION_CRITERIA_H_

#include "../class/optimization_settings.h"


template<typename F>
F computeTheError(const F fval, const F fval_prev,const optimization_settings* settings) {
	if (fval_prev > fval+1E-2 && settings->verbose)
		printf("Error dedected: %1.16f < %1.16f\n", fval_prev, fval);

	return (abs(fval - fval_prev) / fval_prev);
}

template<typename F>
bool termination_criteria(const F error, int it,
		const optimization_settings* settings) {
	//FIXME CHECK
	if (it > 0 && error < settings->toll)
		return true;
	return false;
}

template<typename F>
bool termination_criteria(const F fval, const F fval_prev, int it,
		const optimization_settings* settings) {
	//FIXME CHECK
	if (it > 0 && computeTheError(fval, fval_prev) < settings->toll)
		return true;
	return false;
}

#endif /* TERMINATION_CRITERIA_H_ */
