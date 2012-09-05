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
	return (abs(i) < abs(j));
}

template<typename F>
void vector_sgn(F * y, unsigned int n) { // compute y=sgn(y)
#ifdef _OPENMP
#pragma omp parallel for
#endif various
	for (unsigned int i = 0; i < n; i++) {
		y[i] = sgn(y[i]);
	}
}

void mySort(double * x, const unsigned int length,
		std::vector<double>& myvector) {
	cblas_vector_copy(length, x, 1, &myvector[0], 1);
	sort(myvector.begin(), myvector.end(), abs_value_comparator<double> );
}

void mySort(float * x, const unsigned int length, std::vector<float>& myvector) {
	cblas_vector_copy(length, x, 1, &myvector[0], 1);
	sort(myvector.begin(), myvector.end(), abs_value_comparator<float> );
}

template<typename F>
F find_hard_treshHolding_parameter_with_sorting(F * x,
		const unsigned int length, const unsigned int k,
		std::vector<F>& myvector) {
	//FIXME
	//	For most real-world cases, parallel sort would be optimal algorithm. However, with higher concurrency levels, larger data-set size or heavier comparator/hasher complexity, parallel buffered sort or parallel radix sort may perform magnitudes better.
	mySort(x, length, myvector);
	return abs(myvector[length - k]);
}

template<typename F>
F find_hard_treshHolding_parameter_without_sorting(F * x,
		const unsigned int length, const unsigned int k,
		std::vector<F>& myvector) {
	set<data_for_sorted_map<F> , data_for_sorted_map_comp<F> > data;
	for (unsigned int j = 0; j < k; j++) {
		data.insert(data_for_sorted_map<F> (abs(x[j]), j));
	}
	//	set<data_for_sorted_map<double> >::iterator it;
	//	cout << "!yset contains:";
	//	for (it = data.begin(); it != data.end(); it++)
	//		cout << " " << (*it).value;
	//	cout << endl;
	F thresh = 0;
	for (unsigned int j = k; j < length; j++) {
		F tmp = abs(x[j]);
		thresh = data.rbegin()->value;
		if (tmp > thresh) {
			data.erase(*data.rbegin());
			data.insert(data_for_sorted_map<F> (tmp, j));
		}
	}
	return data.rbegin()->value;
}

template<typename F>
F k_hard_tresholding(F * x, const unsigned int length, const unsigned int k,
		std::vector<F>& myvector, optimization_settings* settings) {
	F treshHold;
	if (settings->hard_tresholding_using_sort) {
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

// Soft treshholding  x_i = (|x_i| - w)_+ sgn(x_i)
template<typename F>
F soft_tresholding(F * x, const unsigned int length,
		const unsigned int constrain, std::vector<F>& myvector) {
	F w = sqrt(constrain);
	mySort(x, length, myvector);
	F lambda_Low = 0;
	F lambda_High = abs(myvector[length - 1]);//lambda = \|x\|_\infty
	F sum_abs_x = 0;
	F sum_abs_x2 = 0;
	F tmp = abs(myvector[length - 1]);
	F subgrad = 0;
	int total_elements = 0;
	for (unsigned int i = 0; i < length; i++) {
		tmp = abs(myvector[length - i - 1]);
		sum_abs_x += tmp;
		sum_abs_x2 += tmp * tmp;
		total_elements++;
		//compute subgradient close to lambda=x_{(i)}
		lambda_Low = tmp;
		subgrad = (sum_abs_x - total_elements * tmp) / sqrt(
				sum_abs_x2 - 2 * tmp * sum_abs_x + total_elements * tmp * tmp);
		if (subgrad >= w) {
			sum_abs_x -= tmp;
			sum_abs_x2 -= tmp * tmp;
			total_elements--;
			break;
		}
		lambda_High = tmp;
	}
	// solve quadratic
	F a = total_elements * (total_elements - w * w);
	F b = 2 * sum_abs_x * w * w - 2 * total_elements * sum_abs_x;
	F c = sum_abs_x * sum_abs_x - w * w * sum_abs_x2;
	w = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
#ifdef DEBUG
	if (w < lambda_Low || w > lambda_High) {
		printf("Problem detected!\n");
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
