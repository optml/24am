//============================================================================
// Name        : GPower.cpp
// Author      : Martin Takac
// Version     :
// Copyright   : GNU
// Description : GPower Method
//============================================================================

#include <iostream>
using namespace std;
#include <stdio.h>
#include "../utils/timer.h"

#include <gsl/gsl_cblas.h>
#include "gpower/sparse_PCA_solver.h"

#include "gpower/optimization_settings.h"
#include "gpower/optimization_statistics.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <cuda.h>
#include "cublas_v2.h"

#include "gpower_cu/gpu_sparse_PCA_solver.h"
#include "gpower/gpower_problem_generator.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

template<typename F>
unsigned int vector_get_nnz(const F * x,const int n) {
	unsigned int nnz = 0;
	for (unsigned int i = 0; i < n; i++) {
		if (x[i] != 0)
		nnz++;
	}
	return nnz;
}

void getFileSize(const char* filename, int& DIM_M, int& DIM_N) {
	FILE * fin = fopen(filename, "r");
	if (fin == NULL) {
	} else {
		fscanf(fin, "%d;%d", &DIM_M, &DIM_N);
		fclose(fin);
	}
}

template<typename F>
void readFromFile(const char* filename, int& DIM_M, int& DIM_N, thrust::host_vector<F> &B) {
	int i, j;
	FILE * fin = fopen(filename, "r");
	if (fin == NULL) {
	} else {
		fscanf(fin, "%d;%d", &DIM_M, &DIM_N);
		for (j = 0; j < DIM_M; j++) {
			for (i = 0; i < DIM_N; i++) {
				float tmp = -1;
				fscanf(fin, "%f;", &tmp);

				float asdf = (float) rand()/RAND_MAX;

				if (asdf>0.5) asdf=-asdf;

				B[IDX2C(j,i,DIM_M)]=tmp;
			}
		}
		fclose(fin);
	}
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
	generateProblem( n, m, h_B,LD_M,LD_N);

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
