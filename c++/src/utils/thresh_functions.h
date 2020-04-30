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
 * tresh_functions.h
 *
 *  Created on: Mar 28, 2012
 *      Author: taki
 */

#ifndef TRESH_FUNCTIONS_H_
#define TRESH_FUNCTIONS_H_

#include "../class/optimization_settings.h"
#include "termination_criteria.h"
#include "my_cblas_wrapper.h"
#include <math.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

using namespace std;
#include <map>
#include <list>
#include <set>

template<typename F>
class data_for_sorted_map {
public:
	F value;
	unsigned int idx;
	data_for_sorted_map(F _value, unsigned int _idx) {
		idx = _idx;
		value = _value;
	}

};

template<typename F>
class data_for_sorted_map_comp {
public:
	bool operator()(const data_for_sorted_map<F>& lhs,
			const data_for_sorted_map<F>& rhs) const {
		return lhs.value > rhs.value;
	}
};



template<typename F>
bool abs_value_comparator(const F & i, const F &j) {
	return (myabs(i) < myabs(j));
}

template<typename F>
void vector_sgn(F * y, unsigned int n) { // compute y=sgn(y)
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int i = 0; i < n; i++) {
		y[i] = sgn(y[i]);
	}
}

void mySort(double * x, const unsigned int length,
		std::vector<double>& myvector) {
	cblas_vector_copy(length, x, 1, &myvector[0], 1);
	sort(myvector.begin(), myvector.end(), abs_value_comparator<double>);
}

void mySort(float * x, const unsigned int length,
		std::vector<float>& myvector) {
	cblas_vector_copy(length, x, 1, &myvector[0], 1);
	sort(myvector.begin(), myvector.end(), abs_value_comparator<float>);
}

template<typename F>
F find_hard_treshHolding_parameter_with_sorting(F * x,
		const unsigned int length, const unsigned int k,
		std::vector<F>& myvector) {
	//	For most real-world cases, parallel sort would be optimal algorithm. However, with higher concurrency levels, larger data-set size or heavier comparator/hasher complexity, parallel buffered sort or parallel radix sort may perform magnitudes better.
	mySort(x, length, myvector);
	return abs(myvector[length - k]);
}

template<typename F>
F find_hard_treshHolding_parameter_without_sorting(F * x,
		const unsigned int length, const unsigned int k,
		std::vector<F>& myvector) {
	set<data_for_sorted_map<F>, data_for_sorted_map_comp<F> > data;
	for (unsigned int j = 0; j < k; j++) {
		data.insert(data_for_sorted_map<F>(abs(x[j]), j));
	}
	F thresh = 0;
	for (unsigned int j = k; j < length; j++) {
		F tmp = abs(x[j]);
		thresh = data.rbegin()->value;
		if (tmp > thresh) {
			data.erase(*data.rbegin());
			data.insert(data_for_sorted_map<F>(tmp, j));
		}
	}
	return data.rbegin()->value;
}

template<typename F>
F k_hard_thresholding(F * x, const unsigned int length, const unsigned int k,
		std::vector<F>& myvector,
		SolverStructures::OptimizationSettings* optimizationSettings) {
	F treshHold;
	if (optimizationSettings->useSortForHardThresholding) {
		treshHold = find_hard_treshHolding_parameter_with_sorting(x, length, k,
				myvector);
	} else {
		treshHold = find_hard_treshHolding_parameter_without_sorting(x, length,
				k, myvector);
	}
	F norm = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:norm)
#endif
	for (unsigned int i = 0; i < length; i++) {
		F val = x[i];
		if (abs(val) < treshHold) {
			x[i] = 0;
		} else {
			norm += val * val;
		}
	}
	return sqrt(norm);
}

template<typename F>
F compute_V_value(F*x, const unsigned int length, F lambda,
		const int constrain) {
	F value = 0;
	for (int i = 0; i < length; i++) {
		if (abs(x[i]) > lambda) {
			F tmp = (abs(x[i]) - lambda);
			value += (tmp * tmp);
		}
	}
	return lambda * sqrt(constrain + 0.0) + sqrt(value);
}

// Soft treshholding  x_i = (|x_i| - w)_+ sgn(x_i)
template<typename F>
F soft_thresholding(F * x, const unsigned int length,
		const unsigned int constrain, std::vector<F>& myvector,
		SolverStructures::OptimizationSettings* optimizationSettings) {
	F sq_constr = sqrt(constrain + 0.0);
	mySort(x, length, myvector);
	F lambda_Low = 0;
	F lambda_High = abs(myvector[length - 1]); //lambda = \|x\|_\infty
	F linfty=abs(myvector[length - 1]);
	F sum_abs_x = 0;
	F sum_abs_x2 = 0;
	F epsilon = 0.000001;
	F tmp = abs(myvector[length - 1]);
	F subgrad = 0;
	int total_elements = 0;
	F subgradOld = 0;
	F w = 0;
	F subgradRightOld = sq_constr;
	for (unsigned int i = 0; i < length; i++) {
		if (i > 0) {
			subgradOld = subgrad;
		}
		tmp = abs(myvector[length - i - 1]);
		sum_abs_x += tmp;
		sum_abs_x2 += tmp * tmp;
		total_elements++;
		//compute subgradient close to lambda=x_{(i)}
		lambda_High = tmp;
		if (i < length - 1)
			lambda_Low = abs(myvector[length - i - 2]);
		else
			lambda_Low = 0;
		F subgradLeft = (sum_abs_x - total_elements * tmp)
				/ sqrt(
						sum_abs_x2 - 2 * tmp * sum_abs_x
								+ total_elements * tmp * tmp);
		if (subgradLeft > sq_constr && subgradRightOld < sq_constr) {
			w = tmp;
			break;
		}
		F subgradRight = (sum_abs_x - total_elements * lambda_Low)
				/ sqrt(
						sum_abs_x2 - 2 * lambda_Low * sum_abs_x
								+ total_elements * lambda_Low * lambda_Low);
		F a = total_elements * (total_elements - sq_constr * sq_constr);
		F b = 2 * sum_abs_x * sq_constr * sq_constr
				- 2 * total_elements * sum_abs_x;
		F c = sum_abs_x * sum_abs_x - sq_constr * sq_constr * sum_abs_x2;
		w = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		if (w > lambda_Low &&  w < lambda_High ) {
			break;
		}
		w = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		if (w > lambda_Low &&  w < lambda_High && w < linfty- epsilon ) {
			break;
		}
		subgradRightOld = subgradRight;
	}

#ifdef DEBUG
	if (w < lambda_Low || w > lambda_High) {
		if (optimizationSettings->verbose)
		printf("Problem detected!  %f < %f < %f  \n",lambda_Low,w,lambda_High);
	}
#endif
	F norm = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:norm)
	for (unsigned int i = 0; i < length; i++) {
		F tmp = x[i];
		F tmp2 = abs(tmp) - w;
		if (tmp2 > 0) {
			const F val = tmp2 * sgn(tmp);
			x[i]=val;
			norm += val * val;
		} else {
			x[i]=0;
		}
	}
#else
	for (unsigned int i = 0; i < length; i++) {
		F tmp = x[i];
		F tmp2 = abs(tmp) - w;
		if (tmp2 > 0) {
			const F val = tmp2 * sgn(tmp);
			x[i] = val;
			norm += val * val;
		} else {
			x[i] = 0;
		}
	}
#endif
	return sqrt(norm);
}

#endif /* TRESH_FUNCTIONS_H_ */
