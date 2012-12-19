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


#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
using namespace SolverStructures;
//#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "../problem_generators/gpower_problem_generator.h"
#include "../gpugpower/gpu_headers.h"

template<typename F>
void run_solver(OptimizationSettings* optimizationSettings) {
//	double start_wall_time = gettime();
//	std::vector<F> B_mat;
//	unsigned int ldB;
//	unsigned int m;
//	unsigned int n;
//	input_ouput_helper::read_csv_file(B_mat, ldB, m, n, optimizationSettings->dataFilePath);
//	optimization_Statisticsistics* optimizationStatistics = new optimization_Statisticsistics();
//	optimizationStatistics->n = n;
//	const F * B = &B_mat[0];
//
//	std::vector<F> x_vec(n, 0);
//	F * x = &x_vec[0];
//	PCA_solver::denseDataSolver(B, ldB, x, m, n, optimizationSettings, optimizationStatistics);
//	double end_wall_time = gettime();
//	optimizationStatistics->totalElapsedTime=end_wall_time-start_wall_time;
//	input_ouput_helper::saveSolverStatistics_and_results(optimizationStatistics, optimizationSettings,x_vec);

}

int mainX(int argc, char *argv[]) {
	OptimizationSettings* optimizationSettings = new OptimizationSettings();
	int optimizationStatisticsus = parseConsoleOptions(optimizationSettings, argc, argv);
	if (optimizationStatisticsus > 0)
		return optimizationStatisticsus;
	if (optimizationSettings->double_precission) {
		run_solver<double>(optimizationSettings);
	} else {
		run_solver<float>(optimizationSettings);
	}
	return 0;
}









template<typename F>
int runTest() {
	OptimizationStatistics* optimizationStatistics = new OptimizationStatistics();
	OptimizationSettings* optimizationSettings = new OptimizationSettings();
	optimizationSettings->maximumIterations = 5;
	optimizationSettings->toll = 0.0001;
	optimizationSettings->totalStartingPoints = 1024;

	mytimer* mt = new mytimer();
	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp,0);
	optimizationSettings->gpu_sm_count=dp.multiProcessorCount;
	optimizationSettings->gpu_max_threads= dp.maxThreadsPerBlock;

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

	cublasStatust optimizationStatisticsus;
	cublasHandle_t handle;
	optimizationStatisticsus = cublasCreate(&handle);
	if (optimizationStatisticsus != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	} else {
		printf("CUBLAS initialized.\n");
	}

	optimizationSettings->penalty=penalty;
	optimizationSettings->constrain=10;
	//	//==================  PENALIZED

	optimizationSettings->algorithm = L0_penalized_L2_PCA;
	mt->start(); denseDataSolver(handle,m, n, d_B, h_x, optimizationSettings, optimizationStatistics,LD_M,LD_N);mt->end();
	nnz = vector_get_nnz(&h_x[0],n);
	printf("FVAL:%f,nnz:%d,%f\n",optimizationStatistics->fval,nnz,mt->getElapsedWallClockTime());

	//-----------------------
	//	optimizationSettings->algorithm = L1_penalized_L2_PCA;
	//	denseDataSolver(m, n, h_B, h_x, optimizationSettings, optimizationStatistics);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",optimizationStatistics->fval,nnz);
	//
	//	//-----------------------
	//	optimizationSettings->algorithm = L0_penalized_L1_PCA;
	//	denseDataSolver(m, n, h_B, h_x, optimizationSettings, optimizationStatistics);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",optimizationStatistics->fval,nnz);
	//
	//	//-----------------------
	//	optimizationSettings->algorithm = L1_penalized_L1_PCA;
	//	denseDataSolver(m, n, h_B, h_x, optimizationSettings, optimizationStatistics);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",optimizationStatistics->fval,nnz);

	//==================  CONSTRAINED
	optimizationSettings->algorithm = L0_constrained_L1_PCA;

	optimizationSettings->useKSelectionAlgorithmGPU=false;
	mt->start(); denseDataSolver(handle,m, n, d_B, h_x, optimizationSettings, optimizationStatistics,LD_M,LD_N);mt->end();
	nnz = vector_get_nnz(&h_x[0],n);
	printf("FVAL:%f,nnz:%d,%f\n",optimizationStatistics->fval,nnz,mt->getElapsedWallClockTime());

	optimizationSettings->useKSelectionAlgorithmGPU=true;
	mt->start(); denseDataSolver(handle,m, n, d_B, h_x, optimizationSettings, optimizationStatistics,LD_M,LD_N);mt->end();
	nnz = vector_get_nnz(&h_x[0],n);
	printf("FVAL:%f,nnz:%d,%f\n",optimizationStatistics->fval,nnz,mt->getElapsedWallClockTime());

	//	optimizationSettings->algorithm = L0_constrained_L2_PCA;
	//	denseDataSolver(m, n, h_B, h_x, optimizationSettings, optimizationStatistics);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",optimizationStatistics->fval,nnz);
	//
	//	optimizationSettings->algorithm = L1_constrained_L1_PCA;
	//	denseDataSolver(m, n, h_B, h_x, optimizationSettings, optimizationStatistics);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",optimizationStatistics->fval,nnz);
	//
	//	optimizationSettings->algorithm = L1_constrained_L2_PCA;
	//	denseDataSolver(m, n, h_B, h_x, optimizationSettings, optimizationStatistics);
	//	nnz = vector_get_nnz(&h_x[0],n);
	//	printf("FVAL:%f,nnz:%d\n",optimizationStatistics->fval,nnz);


	optimizationStatisticsus = cublasDestroy(handle);
	if (optimizationStatisticsus != CUBLAS_STATUS_SUCCESS) {
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
