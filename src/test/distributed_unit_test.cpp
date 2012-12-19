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

#include <stdio.h>
#include <stdlib.h>

#include "../dgpower/distributed_PCA_solver.h"

#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"

using namespace SolverStructures;
#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"

template<typename F>
void test_solver(SolverStructures::OptimizationSettings * optimizationSettings,
		char* multicoreDataset, char* multicoreResult) {
	SolverStructures::OptimizationStatistics* optimizationStatistics =
			new OptimizationStatistics();
	MKL_INT iam, nprocs;
	blacs_pinfo_(&iam, &nprocs);
	double start_all = gettime();
	SPCASolver::DistributedClasses::OptimizationData<F> optimizationDataInstance;
	SPCASolver::DistributedSolver::loadDataFrom2DFilesAndDistribute<F>(
			optimizationDataInstance, optimizationSettings, optimizationStatistics);

	std::vector<F> B_mat;
	unsigned int ldB;
	unsigned int m;
	unsigned int n;
	InputOuputHelper::readCSVFile(B_mat, ldB, m, n, multicoreDataset);
	OptimizationStatistics* optimizationStatistics2 = new OptimizationStatistics();
	optimizationStatistics2->n = n;
	const F * B = &B_mat[0];
	std::vector<F> x_vec(n, 0);
	F * x = &x_vec[0];

	std::vector<SolverStructures::SPCA_Algorithm> algorithms(8);
	algorithms[0] = SolverStructures::L0_penalized_L1_PCA;
	algorithms[1] = SolverStructures::L0_penalized_L2_PCA;
	algorithms[2] = SolverStructures::L1_penalized_L1_PCA;
	algorithms[3] = SolverStructures::L1_penalized_L2_PCA;
	algorithms[4] = SolverStructures::L0_constrained_L1_PCA;
	algorithms[5] = SolverStructures::L0_constrained_L2_PCA;
	algorithms[6] = SolverStructures::L1_constrained_L1_PCA;
	algorithms[7] = SolverStructures::L1_constrained_L2_PCA;
	char* resultDistributed = optimizationSettings->result_file;
	for (int al = 0; al < 8; al++) {
		optimizationSettings->algorithm = algorithms[al];
		SPCASolver::DistributedSolver::denseDataSolver(
				optimizationDataInstance, optimizationSettings, optimizationStatistics);
		if (optimizationSettings->proccess_node == 0) {
			SPCASolver::MulticoreSolver::denseDataSolver(B, ldB, x, m, n, optimizationSettings, optimizationStatistics2);
			optimizationSettings->result_file=multicoreResult;
			InputOuputHelper::save_results(optimizationStatistics, optimizationSettings, x, n);
			cout << "Test " << al << " " << optimizationSettings->algorithm << " "
					<< optimizationStatistics->fval << "  " << optimizationStatistics2->fval << endl;
		}
	}

	/*
	 * STORE RESULT INTO FILE
	 */
	optimizationSettings->result_file=resultDistributed;
	SPCASolver::DistributedSolver::gatherAndStoreBestResultToOutputFile(
			optimizationDataInstance, optimizationSettings, optimizationStatistics);
	if (iam == 0) {
		optimizationStatistics->totalElapsedTime = gettime() - start_all;
		InputOuputHelper::save_OptimizationStatistics(optimizationStatistics, optimizationSettings);
	}

	blacs_gridexit_(&optimizationDataInstance.params.ictxt);
}

int main(int argc, char *argv[]) {
	SolverStructures::OptimizationSettings* optimizationSettings =
			new OptimizationSettings();
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &optimizationSettings->proccess_node);

	optimizationSettings->data_file = "datasets/distributed.dat.";
	optimizationSettings->result_file = "results/distributed_unittest.txt";
	char* multicoreDataset = "datasets/distributed.dat.all";
	char* multicoreResult = "results/distributed_unittest_multicore.txt";
	optimizationSettings->distributed_row_grid_file = 2;
	optimizationSettings->verbose = false;
	optimizationSettings->batch_size = 64;
	optimizationSettings->starting_points = 64;
	optimizationSettings->constrain = 20;
	optimizationSettings->toll = 0.001;
	optimizationSettings->maximumIterations = 100;
	if (optimizationSettings->proccess_node == 0)
		cout << "Double test" << endl;
	test_solver<double>(optimizationSettings, multicoreDataset, multicoreResult);
	MPI_Finalize();
	return 0;
}

