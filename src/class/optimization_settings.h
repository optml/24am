/*
 * optimization_settings.h
 *
 *  Created on: Mar 26, 2012
 *      Author: taki
 */

#ifndef OPTIMIZATION_SETTINGS_H_
#define OPTIMIZATION_SETTINGS_H_

namespace solver_structures {

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



template<typename T> // NOTE: We use typename T instead of std::ostream to make this header-only
T& operator<<(T& stream, SparsePCA_Algorithm& algo) {
	switch (algo) {
	case L0_penalized_L1_PCA:
		stream << "L0_penalized_L1_PCA";
		break;
	case L0_penalized_L2_PCA:
		stream << "L0_penalized_L2_PCA";
		break;
	case L1_penalized_L1_PCA:
		stream << "L1_penalized_L1_PCA";
		break;
	case L1_penalized_L2_PCA:
		stream << "L1_penalized_L2_PCA";
		break;
	case L0_constrained_L1_PCA:
		stream << "L0_constrained_L1_PCA";
		break;
	case L0_constrained_L2_PCA:
		stream << "L0_constrained_L2_PCA";
		break;
	case L1_constrained_L1_PCA:
		stream << "L1_constrained_L1_PCA";
		break;
	case L1_constrained_L2_PCA:
		stream << "L1_constrained_L2_PCA";
		break;
	}
	return stream;
}

class optimization_settings {

public:
	int proccess_node;
	int distributed_row_grid_file;
	double toll;
	bool verbose;
	bool hard_tresholding_using_sort;
	bool gpu_use_k_selection_algorithm;
	bool double_precission;
	double penalty;
	unsigned int constrain;
char* data_file;
char* result_file;
	int gpu_sm_count;
	int gpu_max_threads;

	enum SparsePCA_Algorithm algorithm;

	bool get_values_for_all_points;
	bool get_it_for_all_points;
	int max_it;
	int starting_points;
	int batch_size;
	unsigned int number_of_batches;

	bool on_the_fly_generation;

	optimization_settings() {
		distributed_row_grid_file=0;
		proccess_node=0;
		toll = 0.01;
		constrain = 10;
		penalty = 0;
		verbose = false;
		starting_points = 64;
		hard_tresholding_using_sort = false;
		on_the_fly_generation = false;
		max_it = 100;
		get_values_for_all_points = true;
		gpu_use_k_selection_algorithm = true;
		get_it_for_all_points = true;
		double_precission=false;
	}

	bool isConstrainedProblem() {
		if (algorithm == L0_penalized_L1_PCA || algorithm == L0_penalized_L2_PCA
				|| algorithm == L1_penalized_L1_PCA
				|| algorithm == L1_penalized_L2_PCA) {
			return false;
		} else {
			return true;
		}

	}

	bool isL1ConstrainedProblem() {
		if (algorithm == L0_constrained_L1_PCA
				|| algorithm == L0_constrained_L2_PCA) {
			return false;
		} else {
			return true;
		}

	}

	bool isL1PenalizedProblem() {
		if (algorithm == L0_penalized_L1_PCA
				|| algorithm == L0_penalized_L2_PCA) {
			return false;
		} else {
			return true;
		}

	}

	void chceckInputAndModifyIt(unsigned int n){
		if (this->constrain > n){
			this->constrain=n;
		}
		if (this->batch_size == 0) {
			this->batch_size = this->starting_points;
		}
		this->number_of_batches = this->starting_points / this->batch_size;
		if (this->number_of_batches * this->batch_size < this->starting_points)
			this->number_of_batches++;
		this->starting_points = this->number_of_batches * this->batch_size;

	}

};
}
#endif /* OPTIMIZATION_SETTINGS_H_ */
