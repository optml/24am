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

#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "../gpugpower/gpu_sparse_PCA_solver.h"
using namespace SolverStructures;
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "experiment_utils.h"
#include "../problem_generators/gpower_problem_generator.h"

template<typename F>
int test_solver(SolverStructures::OptimizationSettings * settings) {
	SolverStructures::OptimizationStatistics* stat =
			new OptimizationStatistics();

	ofstream fileOut;
	fileOut.open("results/paper_experiment_gpu_speedup.txt");
	ofstream fileOutCPU;
	fileOutCPU.open("results/paper_experiment_gpu_speedup_cpu.txt");

	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	} else {
		printf("CUBLAS initialized.\n");
	}

	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp, 0);
	settings->gpu_sm_count = dp.multiProcessorCount;
	settings->gpu_max_threads = dp.maxThreadsPerBlock;

	mytimer* mt = new mytimer();
	std::vector<F> B_mat;
	std::vector<F> y;
	int multSC = 1;
	for (int mult = multSC; mult <= 64; mult = mult * 2) {
		int m = 100 * mult;
		int n = 1000 * mult;

		const int MEMORY_BANK_FLOAT_SIZE = MEMORY_ALIGNMENT / sizeof(F);
		const unsigned int LD_M = (
				m % MEMORY_BANK_FLOAT_SIZE == 0 ?
						m :
						(m / MEMORY_BANK_FLOAT_SIZE + 1)
								* MEMORY_BANK_FLOAT_SIZE);
		const unsigned int LD_N = (
				n % MEMORY_BANK_FLOAT_SIZE == 0 ?
						n :
						(n / MEMORY_BANK_FLOAT_SIZE + 1)
								* MEMORY_BANK_FLOAT_SIZE);
		thrust::host_vector<F> h_B(LD_M * n, 0);
		generateProblem(n, m, &h_B[0], m, n, false);
		settings->max_it = 100;
		settings->toll = 0;
		settings->penalty = 0.02;
		settings->constrain = n / 100;
		settings->algorithm = L1_penalized_L1_PCA;
		settings->on_the_fly_generation = false;
		settings->gpu_use_k_selection_algorithm = false;
		stat->n = n;
		// move data to DEVICE
		thrust::device_vector<F> d_B = h_B;
		// allocate vector for solution
		thrust::host_vector<F> h_x(n, 0);

		for (settings->starting_points = 1; settings->starting_points <= 256;
				settings->starting_points = settings->starting_points * 16) {
		settings->batch_size = settings->starting_points;
			mt->start();
			SPCASolver::gpu_sparse_PCA_solver(handle, m, n, d_B, h_x, settings,
					stat, LD_M, LD_N);
			mt->end();
			std::vector<F> x(n, 0);
			for (int i = 0; i < n; i++)
				x[i] = h_x[i];
			logTime(fileOut, mt, stat, settings, x, m, n);


			// CPU
			mt->start();
			SPCASolver::MulticoreSolver::denseDataSolver(&h_B[0], LD_M, &x[0], m, n, settings,
								stat);
			mt->end();
			logTime(fileOutCPU, mt, stat, settings, x, m, n);

		}
	}
	fileOutCPU.close();
	fileOut.close();

	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!cublas shutdown error\n");
		return EXIT_FAILURE;
	}
	return 0;
}

int main(int argc, char *argv[]) {
	SolverStructures::OptimizationSettings* settings =
			new OptimizationSettings();
	settings->result_file = "results/gpu_unittest.txt";
	settings->verbose = false;
	settings->starting_points = 1024;
	settings->batch_size = settings->starting_points;
	settings->on_the_fly_generation = false;
	settings->gpu_use_k_selection_algorithm = false;
	settings->constrain = 20;
	settings->toll = 0.0001;
	settings->max_it = 100;
	cout << "Double test" << endl;
	test_solver<double>(settings);
	return 0;
}

