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

#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
using namespace SolverStructures;
#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "../utils/timer.h"
#include "../problem_generators/gpower_problem_generator.h"
#include "experiment_utils.h"
#include "../problem_generators/data_generation.h"
#include "mkl_types.h"
#include "../gpower/sparse_PCA_solver_for_CSC.h"

template<typename F>
void run_experiments(OptimizationSettings* optimizationSettings, const char* filename,
		const char* description, const char* logfilename) {
	OptimizationStatistics* optimizationStatistics = new OptimizationStatistics();
	ofstream fileOut;
	fileOut.open(logfilename);
	mytimer* mt = new mytimer();
	std::vector<F> B_CSC_Vals;
	std::vector < MKL_INT > B_CSC_Row_Id;
	std::vector < MKL_INT > B_CSC_Col_Ptr;
	std::vector<F> means;
	int m = -1;
	int n = -1;
	bool doMean = true;
	load_doc_data(filename, n, m, B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr,
			means, doMean);
	std::vector<F> x;
	x.resize(n);

	for (int i = 0; i < 10; i++) {
		SPCASolver::sparse_PCA_solver_CSC(&B_CSC_Vals[0], &B_CSC_Row_Id[0],
				&B_CSC_Col_Ptr[0], &x[0], m, n, optimizationSettings, optimizationStatistics, doMean,
				&means[0]);
		printDescriptions(&x[0], n, description, optimizationStatistics, fileOut);
		for (int col = 0; col < n; col++) {
			if (x[col] != 0) {
				means[col] = 0;
				for (int r = B_CSC_Col_Ptr[col]; r < B_CSC_Col_Ptr[col + 1];
						r++) {
					B_CSC_Vals[r] = 0;
				}
			}
		}

	}
	fileOut.close();
}

int main(int argc, char *argv[]) {
	OptimizationSettings* optimizationSettings = new OptimizationSettings();

	optimizationSettings->maximumIterations = 50;
	optimizationSettings->toll = 0.0001;
	optimizationSettings->starting_points = 1024;
	optimizationSettings->constrain = 5;
	optimizationSettings->algorithm = L0_constrained_L2_PCA;

//	char* filename = "datasets/docword.nips.txt";
//	char* description = "datasets/vocab.nips.txt";
//	char* logfilename = "results/nips.txt";
//	run_experiments<double>(optimizationSettings, filename, description, logfilename);

	char* filename = "datasets/docword.nytimes.txt";
	char* description = "datasets/vocab.nytimes.txt";
	char* logfilename = "results/nytimes.txt";
	run_experiments<double>(optimizationSettings, filename, description, logfilename);

	filename = "datasets/docword.pubmed.txt";
	description = "datasets/vocab.pubmed.txt";
	logfilename = "results/pubmed.txt";
	run_experiments<double>(optimizationSettings, filename, description, logfilename);

	return 0;
}
