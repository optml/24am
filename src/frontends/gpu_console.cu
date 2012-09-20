/*
//HEADER INFO
 */

//============================================================================
// Name        :
// Author      : Martin Takac
// Version     :
// Copyright   : GNU
// Description : GPower Method
//============================================================================

#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "../gpugpower/gpu_headers.h"


template<typename F>
int load_data_and_run_solver(solver_structures::optimization_settings* settings) {
	mytimer* mt = new mytimer();
	mt->start();
	solver_structures::optimization_statistics* stat =
			new optimization_statistics();
	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp, 0);
	settings->gpu_sm_count = dp.multiProcessorCount;
	settings->gpu_max_threads = dp.maxThreadsPerBlock;

	unsigned int ldB;
	unsigned int m;
	unsigned int n;
	std::vector<F> B_mat;
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
//FIXME
	settings->gpu_use_k_selection_algorithm = true;
	settings->gpu_use_k_selection_algorithm = false;
	PCA_solver::gpu_sparse_PCA_solver(handle, m, n, d_B, h_x, settings, stat,
			LD_M, LD_N);
	mt->end();
	stat->total_elapsed_time = mt->getElapsedWallClockTime();
	input_ouput_helper::save_statistics_and_results(stat, settings, &h_x[0], n);
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!cublas shutdown error\n");
		return EXIT_FAILURE;
	}
	return 0;
}

int main(int argc, char *argv[]) {
	solver_structures::optimization_settings* settings =
			new optimization_settings();
	int status = parse_console_options(settings, argc, argv);
	if (status > 0)
		return status;
	if (settings->double_precission) {
		load_data_and_run_solver<double>(settings);
	} else {
		load_data_and_run_solver<float>(settings);
	}
	return 0;
}

