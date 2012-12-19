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
 *
 *  Created on: Mar 26, 2012
 *      Author: taki
 *
 *
 *  Class used to specify optimizationSettings for solver
 *
 */

#ifndef optimization_settings_H_
#define optimization_settings_H_

namespace SolverStructures {

enum SPCA_Algorithm // Formulation of PCA
{
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
T& operator<<(T& stream, SPCA_Algorithm& algo) {
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

//class used to set optimizationSettings to solver
class OptimizationSettings {
public:
	int proccess_node; // used only for distributed solver. This is set automatically and contains rank of MPI process
	int distributed_row_grid_file; // size of row-grid. used only for distributed solver. See documentation
	double toll; // final tollerance for solver
	bool verbose;
	bool hard_thresholding_using_sort; // determines if hardthresholding should be done by sorting (better for large constrain value)
									  // or by using an sorted map (better for small constrain parameter)
	bool gpu_use_k_selection_algorithm; // use approximate k-selection algorithm (Russel Steinbach, Jeffrey Blanchard, Bradley Gordon, and Toluwaloju Alabi)
	bool double_precission; // determines if one should use "double" or "float"
	double penalty; // value of penalty parameter
	unsigned int constrain; //value of constrain parameter
	char* data_file; //   path to source data file
	char* result_file; // path to file where result and OptimizationStatistics will be used
	int gpu_sm_count; // gpu number of Streaming Processors
	int gpu_max_threads; // gpu - max number of threads
	enum SPCA_Algorithm algorithm; // algorithm which should be used
	bool get_values_for_all_points; // determines if algorithm should store values for all starting points
	bool storeIterationsForAllPoints; // determines if algorithm should store elapsed iterations for all starting points
	int maximumIterations; //max iteration which one starting point can consume
	int starting_points; // number of starting points which algorithm should use
	int batch_size; // size of batch
	unsigned int number_of_batches; // number of bathes - is computer by solver
	bool onTheFlyMethod; // on the fly generation - not applicable for distributed solver

	OptimizationSettings() {
		distributed_row_grid_file = 0;
		proccess_node = 0;
		toll = 0.01;
		constrain = 10;
		penalty = 0;
		verbose = false;
		starting_points = 64;
		hard_thresholding_using_sort = false;
		onTheFlyMethod = false;
		maximumIterations = 100;
		get_values_for_all_points = true;
		gpu_use_k_selection_algorithm = true;
		storeIterationsForAllPoints = true;
		double_precission = false;
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

	// check input optimizationSettings
	void chceckInputAndModifyIt(unsigned int n) {
		if (this->constrain > n) {
			this->constrain = n;
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
#endif /* optimization_settings_H_ */
