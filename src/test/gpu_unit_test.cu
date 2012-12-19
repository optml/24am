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
#include "../gpugpower/denseDataSolver.h"
using namespace SolverStructures;
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"


template<typename F>
int test_solver(SolverStructures::OptimizationSettings * optimizationSettings,
		char* multicoreDataset, char* multicoreResult) {
	SolverStructures::OptimizationStatistics* optimizationStatistics =
			new OptimizationStatistics();
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

	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp, 0);
	optimizationSettings->gpu_sm_count = dp.multiProcessorCount;
	optimizationSettings->gpu_max_threads = dp.maxThreadsPerBlock;

	InputOuputHelper::read_csv_file(B_mat, ldB, m, n, optimizationSettings->dataFilePath);
	optimizationStatistics->n = n;

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

	cublasStatus_t optimizationStatisticsus;
	cublasHandle_t handle;
	optimizationStatisticsus = cublasCreate(&handle);
	if (optimizationStatisticsus != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	} else {
		printf("CUBLAS initialized.\n");
	}
	optimizationSettings->useKSelectionAlgorithmGPU = false;
	std::vector<SolverStructures::SparsePCA_Algorithm> algorithms(8);
	algorithms[0] = SolverStructures::L0_penalized_L1_PCA;
	algorithms[1] = SolverStructures::L0_penalized_L2_PCA;
	algorithms[2] = SolverStructures::L1_penalized_L1_PCA;
	algorithms[3] = SolverStructures::L1_penalized_L2_PCA;
	algorithms[4] = SolverStructures::L0_constrained_L1_PCA;
	algorithms[5] = SolverStructures::L0_constrained_L2_PCA;
	algorithms[6] = SolverStructures::L1_constrained_L1_PCA;
	algorithms[7] = SolverStructures::L1_constrained_L2_PCA;
	char* resultGPU = optimizationSettings->resultFilePath;
	for (int al = 0; al < 8; al++) {
		optimizationSettings->algorithm = algorithms[al];
		SPCASolver::GPUSolver::denseDataSolver(handle, m, n, d_B, h_x, optimizationSettings,
				optimizationStatistics, LD_M, LD_N);
		optimizationSettings->resultFilePath=resultGPU;
		InputOuputHelper::save_results(optimizationStatistics, optimizationSettings, &h_x[0], n);
		if (optimizationSettings->proccessNode == 0) {
			SPCASolver::MulticoreSolver::denseDataSolver(B, ldB, x, m, n, optimizationSettings, optimizationStatistics2);
			optimizationSettings->resultFilePath = multicoreResult;
			InputOuputHelper::save_results(optimizationStatistics2, optimizationSettings, x, n);
			cout << "Test " << al << " " << optimizationSettings->algorithm << " "
					<< optimizationStatistics->fval << "  " << optimizationStatistics2->fval << endl;
		}
	}
	optimizationStatisticsus = cublasDestroy(handle);
	if (optimizationStatisticsus != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!cublas shutdown error\n");
		return EXIT_FAILURE;
	}
	return 0;
}

int main(int argc, char *argv[]) {
	SolverStructures::OptimizationSettings* optimizationSettings =
			new OptimizationSettings();
	optimizationSettings->resultFilePath = "results/gpu_unittest.txt";
	char* multicoreDataset = "datasets/distributed.dat.all";
	optimizationSettings->dataFilePath = multicoreDataset;
	char* multicoreResult = "results/gpu_unittest_multicore.txt";
	optimizationSettings->verbose = false;
	optimizationSettings->totalStartingPoints = 1024;
	optimizationSettings->batchSize = optimizationSettings->totalStartingPoints;
	optimizationSettings->onTheFlyMethod=false;
	optimizationSettings->useKSelectionAlgorithmGPU=false;
	optimizationSettings->constrain = 20;
	optimizationSettings->toll = 0.0001;
	optimizationSettings->maximumIterations = 100;
	cout << "Double test" << endl;
	test_solver<double>(optimizationSettings, multicoreDataset, multicoreResult);
	return 0;
}

