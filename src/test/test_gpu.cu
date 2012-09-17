//============================================================================
// Name        :
// Author      : Martin Takac
// Version     :
// Copyright   : GNU
// Description : GPower Method
//============================================================================



#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
using namespace solver_structures;
//#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "../problem_generators/gpower_problem_generator.h"
#include "../gpugpower/gpu_headers.h"

template<typename F>
void run_solver(optimization_settings* settings) {
//	double start_wall_time = gettime();
//	std::vector<F> B_mat;
//	unsigned int ldB;
//	unsigned int m;
//	unsigned int n;
//	input_ouput_helper::read_csv_file(B_mat, ldB, m, n, settings->data_file);
//	optimization_statistics* stat = new optimization_statistics();
//	stat->n = n;
//	const F * B = &B_mat[0];
//
//	std::vector<F> x_vec(n, 0);
//	F * x = &x_vec[0];
//	PCA_solver::dense_PCA_solver(B, ldB, x, m, n, settings, stat);
//	double end_wall_time = gettime();
//	stat->total_elapsed_time=end_wall_time-start_wall_time;
//	input_ouput_helper::save_statistics_and_results(stat, settings,x_vec);

}

int mainX(int argc, char *argv[]) {
	optimization_settings* settings = new optimization_settings();
	int status = parse_console_options(settings, argc, argv);
	if (status > 0)
		return status;
	if (settings->double_precission) {
		run_solver<double>(settings);
	} else {
		run_solver<float>(settings);
	}
	return 0;
}









template<typename F>
int runTest() {
	optimization_statistics* stat = new optimization_statistics();
	optimization_settings* settings = new optimization_settings();
	settings->max_it = 5;
	settings->toll = 0.0001;
	settings->starting_points = 1024;

	mytimer* mt = new mytimer();
	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp,0);
	settings->gpu_sm_count=dp.multiProcessorCount;
	settings->gpu_max_threads= dp.maxThreadsPerBlock;

	int m = 501;
	int n = 50001;

	double size=(double)n*m*sizeof(F)/(1024*1024*1024);
	printf("SIZE %f  %dx%d\n",size,m,n);
	const int MEMORY_BANK_FLOAT_SIZE = MEMORY_ALIGNMENT / sizeof(F);
	const unsigned int LD_M = (m%MEMORY_BANK_FLOAT_SIZE == 0? m: (m/MEMORY_BANK_FLOAT_SIZE+1)*MEMORY_BANK_FLOAT_SIZE);
	const unsigned int LD_N = (n%MEMORY_BANK_FLOAT_SIZE == 0? n: (n/MEMORY_BANK_FLOAT_SIZE+1)*MEMORY_BANK_FLOAT_SIZE);



	thrust::host_vector<F> h_B(LD_M*n);
	generateProblem( n, m, &h_B[0],LD_M,LD_N);

	thrust::host_vector<F> h_x(n,0);

	F penalty=0.0001;
	int nnz=0;
	thrust::device_vector<F> d_B=h_B;

	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	} else {
		printf("CUBLAS initialized.\n");
	}

	settings->penalty=penalty;
	settings->constrain=10;
	//	//==================  PENALIZED

	settings->algorithm = L0_penalized_L2_PCA;
	mt->start(); gpu_sparse_PCA_solver(handle,m, n, d_B, h_x, settings, stat,LD_M,LD_N);mt->end();
	nnz = vector_get_nnz(&h_x[0],n);
	printf("FVAL:%f,nnz:%d,%f\n",stat->fval,nnz,mt->getElapsedWallClockTime());

	//-----------------------
	//	settings->algorithm = L1_penalized_L2_PCA;
	//	gpu_sparse_PCA_solver(m, n, h_B, h_x, settings, stat);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",stat->fval,nnz);
	//
	//	//-----------------------
	//	settings->algorithm = L0_penalized_L1_PCA;
	//	gpu_sparse_PCA_solver(m, n, h_B, h_x, settings, stat);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",stat->fval,nnz);
	//
	//	//-----------------------
	//	settings->algorithm = L1_penalized_L1_PCA;
	//	gpu_sparse_PCA_solver(m, n, h_B, h_x, settings, stat);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",stat->fval,nnz);

	//==================  CONSTRAINED
	settings->algorithm = L0_constrained_L1_PCA;

	settings->gpu_use_k_selection_algorithm=false;
	mt->start(); gpu_sparse_PCA_solver(handle,m, n, d_B, h_x, settings, stat,LD_M,LD_N);mt->end();
	nnz = vector_get_nnz(&h_x[0],n);
	printf("FVAL:%f,nnz:%d,%f\n",stat->fval,nnz,mt->getElapsedWallClockTime());

	settings->gpu_use_k_selection_algorithm=true;
	mt->start(); gpu_sparse_PCA_solver(handle,m, n, d_B, h_x, settings, stat,LD_M,LD_N);mt->end();
	nnz = vector_get_nnz(&h_x[0],n);
	printf("FVAL:%f,nnz:%d,%f\n",stat->fval,nnz,mt->getElapsedWallClockTime());

	//	settings->algorithm = L0_constrained_L2_PCA;
	//	gpu_sparse_PCA_solver(m, n, h_B, h_x, settings, stat);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",stat->fval,nnz);
	//
	//	settings->algorithm = L1_constrained_L1_PCA;
	//	gpu_sparse_PCA_solver(m, n, h_B, h_x, settings, stat);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",stat->fval,nnz);
	//
	//	settings->algorithm = L1_constrained_L2_PCA;
	//	gpu_sparse_PCA_solver(m, n, h_B, h_x, settings, stat);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",stat->fval,nnz);


	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!cublas shutdown error\n");
		return EXIT_FAILURE;
	}
	return 0;
}

int main() {

	runTest<float>();

	printf("FLOAT DONE\n");

		runTest<double>();

	//	thrust::host_vector<double> h_z(10, 1);
	//	thrust::device_vector<double> asdf=h_z;

	printf("DOUBLE DONE\n");

	return 0;
}
