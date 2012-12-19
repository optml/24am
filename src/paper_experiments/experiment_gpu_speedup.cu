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
int test_solver(SolverStructures::OptimizationSettings * optimizationSettings) {
	SolverStructures::OptimizationStatisticsistics* optimizationStatistics =
			new OptimizationStatisticsistics();

	ofstream fileOut;
	fileOut.open("results/paper_experiment_gpu_speedup.txt");
	ofstream fileOutCPU;
	fileOutCPU.open("results/paper_experiment_gpu_speedup_cpu.txt");

	cublasoptimizationStatisticsus_t optimizationStatisticsus;
	cublasHandle_t handle;
	optimizationStatisticsus = cublasCreate(&handle);
	if (optimizationStatisticsus != CUBLAS_optimizationStatisticsUS_SUCCESS) {
		fprintf(stderr, "! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	} else {
		printf("CUBLAS initialized.\n");
	}

	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp, 0);
	optimizationSettings->gpu_sm_count = dp.multiProcessorCount;
	optimizationSettings->gpu_max_threads = dp.maxThreadsPerBlock;

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
		optimizationSettings->max_it = 100;
		optimizationSettings->toll = 0;
		optimizationSettings->penalty = 0.02;
		optimizationSettings->constrain = n / 100;
		optimizationSettings->algorithm = L1_penalized_L1_PCA;
		optimizationSettings->onTheFlyMethod = false;
		optimizationSettings->gpu_use_k_selection_algorithm = false;
		optimizationStatistics->n = n;
		// move data to DEVICE
		thrust::device_vector<F> d_B = h_B;
		// allocate vector for solution
		thrust::host_vector<F> h_x(n, 0);

		for (optimizationSettings->starting_points = 1; optimizationSettings->starting_points <= 256;
				optimizationSettings->starting_points = optimizationSettings->starting_points * 16) {
		optimizationSettings->batch_size = optimizationSettings->starting_points;
			mt->start();
			SPCASolver::GPUSolver::gpu_sparse_PCA_solver(handle, m, n, d_B, h_x, optimizationSettings,
					optimizationStatistics, LD_M, LD_N);
			mt->end();
			std::vector<F> x(n, 0);
			for (int i = 0; i < n; i++)
				x[i] = h_x[i];
			logTime(fileOut, mt, optimizationStatistics, optimizationSettings, x, m, n);


			// CPU
			mt->start();
			SPCASolver::MulticoreSolver::denseDataSolver(&h_B[0], LD_M, &x[0], m, n, optimizationSettings,
								optimizationStatistics);
			mt->end();
			logTime(fileOutCPU, mt, optimizationStatistics, optimizationSettings, x, m, n);

		}
	}
	fileOutCPU.close();
	fileOut.close();

	optimizationStatisticsus = cublasDestroy(handle);
	if (optimizationStatisticsus != CUBLAS_optimizationStatisticsUS_SUCCESS) {
		fprintf(stderr, "!cublas shutdown error\n");
		return EXIT_FAILURE;
	}
	return 0;
}

int main(int argc, char *argv[]) {
	SolverStructures::OptimizationSettings* optimizationSettings =
			new OptimizationSettings();
	optimizationSettings->result_file = "results/gpu_unittest.txt";
	optimizationSettings->verbose = false;
	optimizationSettings->starting_points = 1024;
	optimizationSettings->batch_size = optimizationSettings->starting_points;
	optimizationSettings->onTheFlyMethod = false;
	optimizationSettings->gpu_use_k_selection_algorithm = false;
	optimizationSettings->constrain = 20;
	optimizationSettings->toll = 0.0001;
	optimizationSettings->max_it = 100;
	cout << "Double test" << endl;
	test_solver<double>(optimizationSettings);
	return 0;
}

