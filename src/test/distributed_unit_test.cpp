/*
  *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M. Takac and S. Damla Ahipasaoglu 
 *    "Alternating Maximization: Unified Framework and 24 Parallel Codes for L1 and L2 based Sparse PCA"
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

using namespace solver_structures;
#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"

template<typename F>
void test_solver(solver_structures::optimization_settings * settings,
		char* multicoreDataset, char* multicoreResult) {
	solver_structures::optimization_statistics* stat =
			new optimization_statistics();
	MKL_INT iam, nprocs;
	blacs_pinfo_(&iam, &nprocs);
	double start_all = gettime();
	PCA_solver::distributed_solver::optimization_data<F> optimization_data_inst;
	PCA_solver::distributed_solver::load_data_from_2d_files_and_distribution<F>(
			optimization_data_inst, settings, stat);

	std::vector<F> B_mat;
	unsigned int ldB;
	unsigned int m;
	unsigned int n;
	input_ouput_helper::read_csv_file(B_mat, ldB, m, n, multicoreDataset);
	optimization_statistics* stat2 = new optimization_statistics();
	stat2->n = n;
	const F * B = &B_mat[0];
	std::vector<F> x_vec(n, 0);
	F * x = &x_vec[0];

	std::vector<solver_structures::SparsePCA_Algorithm> algorithms(8);
	algorithms[0] = solver_structures::L0_penalized_L1_PCA;
	algorithms[1] = solver_structures::L0_penalized_L2_PCA;
	algorithms[2] = solver_structures::L1_penalized_L1_PCA;
	algorithms[3] = solver_structures::L1_penalized_L2_PCA;
	algorithms[4] = solver_structures::L0_constrained_L1_PCA;
	algorithms[5] = solver_structures::L0_constrained_L2_PCA;
	algorithms[6] = solver_structures::L1_constrained_L1_PCA;
	algorithms[7] = solver_structures::L1_constrained_L2_PCA;
	char* resultDistributed = settings->result_file;
	for (int al = 0; al < 8; al++) {
		settings->algorithm = algorithms[al];
		PCA_solver::distributed_solver::distributed_sparse_PCA_solver(
				optimization_data_inst, settings, stat);
		if (settings->proccess_node == 0) {
			PCA_solver::dense_PCA_solver(B, ldB, x, m, n, settings, stat2);
			settings->result_file=multicoreResult;
			input_ouput_helper::save_results(stat, settings, x, n);
			cout << "Test " << al << " " << settings->algorithm << " "
					<< stat->fval << "  " << stat2->fval << endl;
		}
	}

	/*
	 * STORE RESULT INTO FILE
	 */
	settings->result_file=resultDistributed;
	PCA_solver::distributed_solver::gather_and_store_best_result_to_file(
			optimization_data_inst, settings, stat);
	if (iam == 0) {
		stat->total_elapsed_time = gettime() - start_all;
		input_ouput_helper::save_statistics(stat, settings);
	}

	blacs_gridexit_(&optimization_data_inst.params.ictxt);
}

int main(int argc, char *argv[]) {
	solver_structures::optimization_settings* settings =
			new optimization_settings();
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
	settings->toll = 0.01;
	settings->max_it = 100;
	if (settings->proccess_node == 0)
		cout << "Double test" << endl;
	test_solver<double>(settings, multicoreDataset, multicoreResult);
	MPI_Finalize();
	return 0;
}

