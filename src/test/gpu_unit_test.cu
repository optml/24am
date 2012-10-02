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

#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"

using namespace solver_structures;
#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "../gpugpower/gpu_sparse_PCA_solver.h"

template<typename F>
void test_solver(solver_structures::optimization_settings * settings,
		char* multicoreDataset, char* multicoreResult) {
	solver_structures::optimization_statistics* stat =
			new optimization_statistics();
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

	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp, 0);
	settings->gpu_sm_count = dp.multiProcessorCount;
	settings->gpu_max_threads = dp.maxThreadsPerBlock;

	unsigned int ldB;
	unsigned int m;
	unsigned int n;
	std::vector < F > B_mat;
	input_ouput_helper::read_csv_file(B_mat, ldB, m, n, settings->data_file);
	stat->n = n;

	const int MEMORY_BANK_FLOAT_SIZE = MEMORY_ALIGNMENT / sizeof(F);
	const unsigned int LD_M = (
			m % MEMORY_BANK_FLOAT_SIZE == 0 ?
					m :
					(m / MEMORY_BANK_FLOAT_SIZE + 1) * MEMORY_BANK_FLOAT_SIZE);
	const unsigned int LD_N = (
			n % MEMORY_BANK_FLOAT_SIZE == 0 ?
					n :
					(n / MEMORY_BANK_FLOAT_SIZE + 1) * MEMORY_BANK_FLOAT_SIZE);
	thrust::host_vector<F> h_B(LD_M * n, 0);
	// get data into h_B;
	for (unsigned int row = 0; row < m; row++) {
		for (unsigned int col = 0; col < n; col++) {
			h_B[row + col * LD_M] = B_mat[row + col * m];
		}
	}
	// allocate vector for solution
	thrust::host_vector<F> h_x(n, 0);
	// move data to DEVICE
	thrust::device_vector<F> d_B = h_B;

	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	} else {
		printf("CUBLAS initialized.\n");
	}
	settings->gpu_use_k_selection_algorithm = false;
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
		PCA_solver::gpu_sparse_PCA_solver(handle, m, n, d_B, h_x, settings,
				stat, LD_M, LD_N);
		if (settings->proccess_node == 0) {
			PCA_solver::dense_PCA_solver(B, ldB, x, m, n, settings, stat2);
			settings->result_file = multicoreResult;
			input_ouput_helper::save_results(stat, settings, x, n);
			cout << "Test " << al << " " << settings->algorithm << " "
					<< stat->fval << "  " << stat2->fval << endl;
		}
	}
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!cublas shutdown error\n");
		return EXIT_FAILURE;
	}
}

int main(int argc, char *argv[]) {
	solver_structures::optimization_settings* settings =
			new optimization_settings();
	settings->data_file = "datasets/distributed.dat.";
	settings->result_file = "results/gpu_unittest.txt";
	char* multicoreDataset = "datasets/distributed.dat.all";
	char* multicoreResult = "results/gpu_unittest_multicore.txt";
	settings->verbose = false;
	settings->batch_size = 64;
	settings->starting_points = 64;
	settings->constrain = 20;
	settings->toll = 0.001;
	settings->max_it = 100;
	cout << "Double test" << endl;
	test_solver<double>(settings, multicoreDataset, multicoreResult);
	return 0;
}

