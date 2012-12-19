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
void test_solver(SolverStructures::OptimizationSettings * settings,
		char* multicoreDataset, char* multicoreResult) {
	SolverStructures::OptimizationStatistics* stat =
			new OptimizationStatistics();
	MKL_INT iam, nprocs;
	blacs_pinfo_(&iam, &nprocs);
	double start_all = gettime();
	SPCASolver::DistributedClasses::OptimizationData<F> optimization_data_inst;
	SPCASolver::DistributedSolver::loadDataFrom2DFilesAndDistribute<F>(
			optimization_data_inst, settings, stat);

	std::vector<F> B_mat;
	unsigned int ldB;
	unsigned int m;
	unsigned int n;
	InputOuputHelper::readCSVFile(B_mat, ldB, m, n, multicoreDataset);
	OptimizationStatistics* stat2 = new OptimizationStatistics();
	stat2->n = n;
	const F * B = &B_mat[0];
	std::vector<F> x_vec(n, 0);
	F * x = &x_vec[0];

	std::vector<SolverStructures::SparsePCA_Algorithm> algorithms(8);
	algorithms[0] = SolverStructures::L0_penalized_L1_PCA;
	algorithms[1] = SolverStructures::L0_penalized_L2_PCA;
	algorithms[2] = SolverStructures::L1_penalized_L1_PCA;
	algorithms[3] = SolverStructures::L1_penalized_L2_PCA;
	algorithms[4] = SolverStructures::L0_constrained_L1_PCA;
	algorithms[5] = SolverStructures::L0_constrained_L2_PCA;
	algorithms[6] = SolverStructures::L1_constrained_L1_PCA;
	algorithms[7] = SolverStructures::L1_constrained_L2_PCA;
	char* resultDistributed = settings->result_file;
	for (int al = 0; al < 8; al++) {
		settings->algorithm = algorithms[al];
		SPCASolver::DistributedSolver::denseDataSolver(
				optimization_data_inst, settings, stat);
		if (settings->proccess_node == 0) {
			SPCASolver::MulticoreSolver::denseDataSolver(B, ldB, x, m, n, settings, stat2);
			settings->result_file=multicoreResult;
			InputOuputHelper::save_results(stat, settings, x, n);
			cout << "Test " << al << " " << settings->algorithm << " "
					<< stat->fval << "  " << stat2->fval << endl;
		}
	}

	/*
	 * STORE RESULT INTO FILE
	 */
	settings->result_file=resultDistributed;
	SPCASolver::DistributedSolver::gather_and_store_best_result_to_file(
			optimization_data_inst, settings, stat);
	if (iam == 0) {
		stat->total_elapsed_time = gettime() - start_all;
		InputOuputHelper::save_statistics(stat, settings);
	}

	blacs_gridexit_(&optimization_data_inst.params.ictxt);
}

int main(int argc, char *argv[]) {
	SolverStructures::OptimizationSettings* settings =
			new OptimizationSettings();
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &settings->proccess_node);

	settings->data_file = "datasets/distributed.dat.";
	settings->result_file = "results/distributed_unittest.txt";
	char* multicoreDataset = "datasets/distributed.dat.all";
	char* multicoreResult = "results/distributed_unittest_multicore.txt";
	settings->distributed_row_grid_file = 2;
	settings->verbose = false;
	settings->batch_size = 64;
	settings->starting_points = 64;
	settings->constrain = 20;
	settings->toll = 0.001;
	settings->max_it = 100;
	if (settings->proccess_node == 0)
		cout << "Double test" << endl;
	test_solver<double>(settings, multicoreDataset, multicoreResult);
	MPI_Finalize();
	return 0;
}

