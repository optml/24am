/*
 * optimization_statistics.h
 *
 *  Created on: Mar 29, 2012
 *      Author: taki
 */

#ifndef OPTIMIZATION_STATISTICS_H_
#define OPTIMIZATION_STATISTICS_H_

#include <vector>

class optimization_statistics {

public:
	unsigned int it;
	double fval;
	double error;
	double true_computation_time;
	std::vector<double> values;
	std::vector<int> iters;
	std::vector<int> cardinalities;
	optimization_statistics(){
      it=0;
	}
};

#endif /* OPTIMIZATION_STATISTICS_H_ */