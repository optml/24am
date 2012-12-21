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

enum SPCA_Formulation // Formulation of PCA
{
	L0_constrained_L2_PCA = 0,
	L0_constrained_L1_PCA,
	L1_constrained_L2_PCA,
	L1_constrained_L1_PCA,
	L0_penalized_L2_PCA,
	L0_penalized_L1_PCA,
	L1_penalized_L2_PCA,
	L1_penalized_L1_PCA
};

template<typename T> // NOTE: We use typename T instead of std::ostream to make this header-only
T& operator<<(T& stream, SPCA_Formulation& formulation) {
	switch (formulation) {
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
	int proccessNode; // used only for distributed solver. This is set automatically and contains rank of MPI process
	int distributedRowGridFile; // size of row-grid. used only for distributed solver. See documentation
	double toll; // final tollerance for solver
	bool verbose;
	bool useSortForHardThresholding; // determines if hardthresholding should be done by sorting (better for large constrain value)
									 // or by using an sorted map (better for small constrain parameter)
	bool useKSelectionAlgorithmGPU; // use approximate k-selection algorithm (Russel Steinbach, Jeffrey Blanchard, Bradley Gordon, and Toluwaloju Alabi)
	bool useDoublePrecision; // determines if one should use "double" or "float"
	double penalty; // value of penalty parameter
	unsigned int constrain; //value of constrain parameter
	char* dataFilePath; //   path to source data file
	char* resultFilePath; // path to file where result and OptimizationStatistics will be used
	int gpu_sm_count; // gpu number of Streaming Processors
	int gpu_max_threads; // gpu - max number of threads
	enum SPCA_Formulation formulation; // formulation which should be used
	bool getValuesForAllStartingPoints; // determines if algorithm should store values for all starting points
	bool storeIterationsForAllPoints; // determines if algorithm should store elapsed iterations for all starting points
	int maximumIterations; //max iteration which one starting point can consume
	int totalStartingPoints; // number of starting points which algorithm should use
	int batchSize; // size of batch
	unsigned int totalBatches; // number of bathes - is computer by solver
	bool useOTF; // on the fly generation - not applicable for distributed solver

	OptimizationSettings() {
		distributedRowGridFile = 0;
		proccessNode = 0;
		toll = 0.01;
		constrain = 10;
		penalty = 0;
		verbose = false;
		totalStartingPoints = 64;
		batchSize = 64;
		useSortForHardThresholding = false;
		useOTF = false;
		maximumIterations = 100;
		getValuesForAllStartingPoints = true;
		useKSelectionAlgorithmGPU = true;
		storeIterationsForAllPoints = true;
		useDoublePrecision = false;
	}

	bool isConstrainedProblem() {
		if (this->formulation == L0_penalized_L1_PCA || this->formulation== L0_penalized_L2_PCA
				|| this->formulation == L1_penalized_L1_PCA
				|| this->formulation == L1_penalized_L2_PCA) {
			return false;
		} else {
			return true;
		}
	}

	bool isL1ConstrainedProblem() {
		if (this->formulation == L0_constrained_L1_PCA
				|| this->formulation== L0_constrained_L2_PCA) {
			return false;
		} else {
			return true;
		}
	}

	bool isL1PenalizedProblem() {
		if (this->formulation == L0_penalized_L1_PCA
				|| this->formulation == L0_penalized_L2_PCA) {
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
		if (this->batchSize == 0) {
			this->batchSize = this->totalStartingPoints;
		}
		this->totalBatches = this->totalStartingPoints / this->batchSize;
		if (this->totalBatches * this->batchSize < this->totalStartingPoints)
			this->totalBatches++;
		this->totalStartingPoints = this->totalBatches * this->batchSize;

	}

};
}
#endif /* optimization_settings_H_ */
