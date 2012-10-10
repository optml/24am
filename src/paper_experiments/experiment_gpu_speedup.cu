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
#include "gpower/timer.h"

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
#include "gpower_cu/gpu_sparse_PCA_solver_single.h"
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
				//				float asdf = (float) rand()/RAND_MAX;
				//				if (asdf>0.5) asdf=-asdf;
				B[IDX2C(j,i,DIM_M)]=tmp;
			}
		}
		fclose(fin);
	}
}

FILE * global_fin;

void logTime(int GPU, int nnz, mytimer* mt, optimization_statistics* stat,
		optimization_settings* settings, int n, int m, int sizeofvariable) {
	printf("%d,%d,%1.5f,%d,%f,%f,%d,%d,%d,%d,%d\n", settings->algorithm, GPU,
			stat->fval, nnz, mt->getElapsedCPUTime(),
			mt->getElapsedWallClockTime(), stat->it, n, m,
			settings->starting_points, sizeofvariable);
	fprintf(global_fin, "%d,%d,%1.5f,%d,%f,%f,%d,%d,%d,%d,%d\n",
			settings->algorithm, GPU, stat->fval, nnz, mt->getElapsedCPUTime(),
			mt->getElapsedWallClockTime(), stat->it, n, m,
			settings->starting_points, sizeofvariable);
	fflush(global_fin);

}


template<typename F>
int runTest() {

	mytimer* mt = new mytimer();
	optimization_statistics* stat = new optimization_statistics();
	optimization_settings* settings = new optimization_settings();
	settings->max_it = 50;
	settings->toll = 0.000;

	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	} else {
		printf("CUBLAS initialized.\n");
	}

	thrust::host_vector<F> h_B;
	thrust::host_vector<F> h_x;

	SparsePCA_Algorithm algorithms[8];

	algorithms[0]=L0_penalized_L1_PCA;
	algorithms[1]= L0_penalized_L2_PCA;
	algorithms[2]=L1_penalized_L1_PCA;
	algorithms[3]=L1_penalized_L2_PCA;
	algorithms[4]=L0_constrained_L1_PCA;
	algorithms[5]=L0_constrained_L2_PCA;
	algorithms[6]=L1_constrained_L1_PCA;
	algorithms[7]=L1_constrained_L2_PCA;
	int nn=1024;
	for (int i=0;i<9;i++) {

		int n=nn;
		int m=n/100;
		n=n*2;
		double size=(double)n*m*sizeof(F)/(1024*1024*1024);
		printf("SIZE %f\n",size);
		h_B.resize(m*n);
		h_x.resize(n);

		generateProblem( n, m, h_B);
		thrust::device_vector<F> d_B = h_B;

		settings->constrain = 100;
		settings->penalty = 0.1;

		for (int sp=1;sp<=256;sp=sp*2) {
			settings->starting_points = sp;
			settings->batch_size=sp;
			for (int alg=2;alg<8;alg=alg+4) {
//			for (int alg=0;alg<8;alg++) {

				settings->algorithm = algorithms[alg];
				mt->start();gpu_sparse_PCA_solver(handle,m, n, d_B, h_x, settings, stat,m,n);mt->end();
				logTime(1, vector_get_nnz(&h_x[0],n), mt,stat,settings,n,m,sizeof(F));

//						mt->start();gpu_sparse_PCA_solver_single(handle,m, n, d_B, h_x, settings, stat);mt->end();
//						logTime(1, vector_get_nnz(&h_x[0],n), mt,stat,settings,n,m);


//				if (sp<=256) {
					mt->start();
					stat->fval = sparse_PCA_solver(&h_B[0], m, &h_x[0],m, n,
							settings, stat);
					mt->end();
					logTime(0, vector_get_nnz(&h_x[0],n), mt,stat,settings,n,m,sizeof(F));
//				}
				printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
			}

		}
		nn=nn*2;
	}
	//	//-----------------------
	//	settings->algorithm = L1_penalized_L2_PCA;
	//	mt->start();gpu_sparse_PCA_solver(handle,m, n, h_B, h_x, settings, stat);mt->end();
	//	logTime(1, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	mt->start();
	//	stat->fval = sparse_PCA_solver(&h_B[0], m, &h_x[0],m, n,
	//			settings, stat);
	//	mt->end();
	//	logTime(0, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
	//
	//	//-----------------------
	//	settings->algorithm = L0_penalized_L1_PCA;
	//	mt->start();gpu_sparse_PCA_solver(handle,m, n, h_B, h_x, settings, stat);mt->end();
	//	logTime(1, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	mt->start();
	//	stat->fval = sparse_PCA_solver(&h_B[0], m, &h_x[0],m, n,
	//			settings, stat);
	//	mt->end();
	//	logTime(0, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
	//
	//	//-----------------------
	//	settings->algorithm = L1_penalized_L1_PCA;
	//	mt->start();gpu_sparse_PCA_solver(handle,m, n, h_B, h_x, settings, stat);mt->end();
	//	logTime(1, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	mt->start();
	//	stat->fval = sparse_PCA_solver(&h_B[0], m, &h_x[0],m, n,
	//			settings, stat);
	//	mt->end();
	//	logTime(0, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
	//
	//	//==================  CONSTRAINED
	//	settings->algorithm = L0_constrained_L1_PCA;
	//	mt->start();gpu_sparse_PCA_solver(handle,m, n, h_B, h_x, settings, stat);mt->end();
	//	logTime(1, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	mt->start();
	//	stat->fval = sparse_PCA_solver(&h_B[0], m, &h_x[0],m, n,
	//			settings, stat);
	//	mt->end();
	//	logTime(0, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
	//
	//	settings->algorithm = L0_constrained_L2_PCA;
	//	mt->start();gpu_sparse_PCA_solver(handle,m, n, h_B, h_x, settings, stat);mt->end();
	//	logTime(1, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	mt->start();
	//	stat->fval = sparse_PCA_solver(&h_B[0], m, &h_x[0],m, n,
	//			settings, stat);
	//	mt->end();
	//	logTime(0, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
	//
	//	settings->algorithm = L1_constrained_L1_PCA;
	//	mt->start();gpu_sparse_PCA_solver(handle,m, n, h_B, h_x, settings, stat);mt->end();
	//	logTime(1, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	mt->start();
	//	stat->fval = sparse_PCA_solver(&h_B[0], m, &h_x[0],m, n,
	//			settings, stat);
	//	mt->end();
	//	logTime(0, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
	//
	//	settings->algorithm = L1_constrained_L2_PCA;
	//	mt->start();gpu_sparse_PCA_solver(handle,m, n, h_B, h_x, settings, stat);mt->end();
	//	logTime(1, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	mt->start();
	//	stat->fval = sparse_PCA_solver(&h_B[0], m, &h_x[0],m, n,
	//			settings, stat);
	//	mt->end();
	//	logTime(0, vector_get_nnz(&h_x[0],n), mt,stat,settings);
	//	printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");

	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!cublas shutdown error\n");
		return EXIT_FAILURE;
	}

	return 0;

}

int main() {
	global_fin = fopen("/exports/home/s1052689/gpu_test2.log", "w");
	runTest<float>();

	printf("DONE\n");
//	runTest<double>();

	fclose(global_fin);

	return 0;
}
