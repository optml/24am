/*
 * optimization_settings.h
 *
 *  Created on: Mar 26, 2012
 *      Author: taki
 */

#ifndef OPTIMIZATION_SETTINGS_H_
#define OPTIMIZATION_SETTINGS_H_

enum SparsePCA_Algorithm {
	L0_penalized_L1_PCA = 0,
	L0_penalized_L2_PCA,
	L1_penalized_L1_PCA,
	L1_penalized_L2_PCA,
	L0_constrained_L1_PCA,
	L0_constrained_L2_PCA,
	L1_constrained_L1_PCA,
	L1_constrained_L2_PCA
};

class optimization_settings {

public:
	double toll;
	bool verbose;
	bool hard_tresholding_using_sort;
	bool gpu_use_k_selection_algorithm;
	double penalty;
	unsigned int constrain;

	int gpu_sm_count;
	int gpu_max_threads;

	enum SparsePCA_Algorithm algorithm;

	bool get_values_for_all_points;
	bool get_it_for_all_points;
	unsigned int max_it;
	unsigned int starting_points;
	unsigned int batch_size;

	bool on_the_fly_generation;

	optimization_settings() {
		toll = 0.01;
		constrain = 1;
		penalty = 0;
		verbose = true;
		starting_points = 1;
		hard_tresholding_using_sort = false;
		on_the_fly_generation = false;
		max_it = 1000;
		get_values_for_all_points = true;
		gpu_use_k_selection_algorithm = true;
		get_it_for_all_points = true;
	}

	bool isConstrainedProblem() {
		if (algorithm == L0_penalized_L1_PCA || algorithm
				== L0_penalized_L2_PCA || algorithm == L1_penalized_L1_PCA
				|| algorithm == L1_penalized_L2_PCA) {
			return false;
		} else {
			return true;
		}

	}

	bool isL1ConstrainedProblem() {
		if (algorithm == L0_constrained_L1_PCA || algorithm
				== L0_constrained_L2_PCA) {
			return false;
		} else {
			return true;
		}

	}

	bool isL1PenalizedProblem() {
		if (algorithm == L0_penalized_L1_PCA || algorithm
				== L0_penalized_L2_PCA) {
			return false;
		} else {
			return true;
		}

	}

};

#endif /* OPTIMIZATION_SETTINGS_H_ */
