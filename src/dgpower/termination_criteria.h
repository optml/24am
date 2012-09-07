/*
 * termination_criteria.h
 *
 *  Created on: Apr 11, 2012
 *      Author: taki
 */

#ifndef TERMINATION_CRITERIA_H_
#define TERMINATION_CRITERIA_H_

#include "optimization_settings.h"

double computeTheError(const double fval, const double fval_prev);

int termination_criteria(const double error, int it, const struct optimization_settings* settings);



#endif /* TERMINATION_CRITERIA_H_ */
