/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M. Takac and S. Damla Ahipasaoglu 
 *    "Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 * 
 */


#include <iostream>
using namespace std;
#include <stdio.h>
#include "../utils/timer.h"
#include "../utils/gsl_helper.h"
#include "../utils/openmp_helper.h"
#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
using namespace SolverStructures ;
#include "../utils/various.h"
#include "../gpower/sparse_PCA_solver.h"
#include "../problem_generators/gpower_problem_generator.h"
//char* filename = "/exports/home/s1052689/costPerSP.dat";
	char* filename = "/tmp/generated.dat.";
char final_file[1000];
////
//
#include "../problem_generators/random_generator.h"
//
//
void logTime(const char* label, double fval, double fval2, int nnz,
		mytimer* mt, OptimizationStatistics* optimizationStatistics, double refval) {
	printf("%s,%1.5f,%1.5f,%d,%f,%f,%d,%f\n", label, fval, fval2, nnz,
			mt->getElapsedCPUTime(), mt->getElapsedWallClockTime(), optimizationStatistics->it,
			refval);
}



FILE * fin;

template<typename F>
int test() {

	init_random_seeds();

	mytimer* mt = new mytimer();
	mt->start();
	//	gsl_matrix * B;
	//	gsl_matrix * BT;
	std::vector<F> h_B;
	std::vector < F > x;

	int mult = 100;
	for (mult = 100; mult <= 4000; mult = mult * 2) {
		int m = mult;
		int n = 10 * mult;
		h_B.resize(m * n);
		x.resize(n);
		//	gsl_vector * x;
		if (true) {
			generateProblem(n, m, &h_B[0]);
			//		B = gsl_matrix_alloc(m, n);
			//		BT = gsl_matrix_alloc(n, m);
			//		x = gsl_vector_alloc(n);
			//		generate_random_instance(n, m, B, BT, x);
		} else {
			//			getFileSize("/document/phd/c/GPower/resources/data.txt0", m, n);
			//			B = gsl_matrix_alloc(m, n);
			//			BT = gsl_matrix_alloc(n, m);
			//			readFromFile("/document/phd/c/GPower/resources/data.txt0", m, n, B,
			//					BT);

			//		x = gsl_vector_alloc(n);
		}

		OptimizationStatistics* optimizationStatistics = new OptimizationStatistics();

		OptimizationSettings* optimizationSettings = new OptimizationSettings();

		optimizationSettings->maximumIterations = 10;
		optimizationSettings->tolerance = 0.;
		optimizationSettings->totalStartingPoints = 1;

		F fval = 0;
		F fval2 = 0;
		F l1_norm;
		unsigned int nnz = 0;
		mt->end();
		cout << "Problem generation took " << mt->getElapsedCPUTime() << " "
				<< mt->getElapsedWallClockTime() << endl;
		//============================
		optimizationSettings->constrain = 10;
		optimizationSettings->penalty = 0.01;
		const F penalty = optimizationSettings->penalty;
		const unsigned int constrain = optimizationSettings->constrain;

		SparsePCA_Algorithm algorithms[8];

		algorithms[0] = L0_penalized_L1_PCA;
		algorithms[1] = L0_penalized_L2_PCA;
		algorithms[2] = L1_penalized_L1_PCA;
		algorithms[3] = L1_penalized_L2_PCA;
		algorithms[4] = L0_constrained_L1_PCA;
		algorithms[5] = L0_constrained_L2_PCA;
		algorithms[6] = L1_constrained_L1_PCA;
		algorithms[7] = L1_constrained_L2_PCA;

		for (int alg = 0; alg < 8; alg++) {
			optimizationSettings->formulation = algorithms[alg];
			for (optimizationSettings->totalStartingPoints = 1; optimizationSettings->totalStartingPoints
					<= 128 * 8*4; //
			optimizationSettings->totalStartingPoints = optimizationSettings->totalStartingPoints * 2+1) {
				mt->start();
				optimizationStatistics->fval = SPCASolver::MulticoreSolver::denseDataSolver(&h_B[0], m, &x[0], m, n,
						optimizationSettings, optimizationStatistics);
				mt->end();
				printf(
						"%d,%d,%f,%f,%d,%d,%d,%d\n",
						alg,
						optimizationSettings->totalStartingPoints,
						optimizationStatistics->totalTrueComputationTime,
						optimizationStatistics->totalTrueComputationTime / (0.0
								+ optimizationSettings->totalStartingPoints * optimizationSettings->maximumIterations),
						m, n, TOTAL_THREADS, sizeof(F));
				fprintf(
						fin,
						"%d,%d,%f,%f,%d,%d,%d,%d\n",
						alg,
						optimizationSettings->totalStartingPoints,
						optimizationStatistics->totalTrueComputationTime,
						optimizationStatistics->totalTrueComputationTime / (0.0
								+ optimizationSettings->totalStartingPoints * optimizationSettings->maximumIterations),
						m, n, TOTAL_THREADS, sizeof(F));

			}
		}
		/*	//----------------- CPU L1 Penalized L1 PCA
		 optimizationSettings->formulation = L1_penalized_L1_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(h_B, m, x->data, B->size1, B->size2,
		 optimizationSettings, optimizationStatistics);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dasum(y) - penalty * gsl_blas_dasum(x);
		 logTime("L1-Pen-L1   ", fval, fval2, nnz, mt, optimizationStatistics,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L1 Penalized L2 PCA
		 optimizationSettings->formulation = L1_penalized_L2_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 optimizationSettings, optimizationStatistics);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dnrm2(y) - penalty * gsl_blas_dasum(x);
		 logTime("L1-Pen-L2 BT", fval, fval2, nnz, mt, optimizationStatistics,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L0 Penalized L1 PCA
		 optimizationSettings->formulation = L0_penalized_L1_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 optimizationSettings, optimizationStatistics);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dasum(y);
		 fval2 = fval2 * fval2 - penalty * nnz;
		 logTime("L0-Pen-L1   ", fval, fval2, nnz, mt, optimizationStatistics,
		 computeReferentialValue(B, x, y));

		 //============= L0 Pen L2
		 optimizationSettings->formulation = L0_penalized_L2_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 optimizationSettings, optimizationStatistics);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dnrm2(y);
		 fval2 = fval2 * fval2 - penalty * nnz;
		 logTime("L0-Pen-L2 M1", fval, fval2, nnz, mt, optimizationStatistics,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L0 Constrained L2 PCA
		 optimizationSettings->formulation = L0_constrained_L2_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 optimizationSettings, optimizationStatistics);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dnrm2(y);
		 logTime("L0-Con-L2   ", fval, fval2, nnz, mt, optimizationStatistics,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L1 Constrained L2 PCA
		 optimizationSettings->formulation = L1_constrained_L2_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 optimizationSettings, optimizationStatistics);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dnrm2(y);
		 l1_norm = gsl_blas_dasum(x);
		 logTime("L1-Con-L2   ", fval, fval2, nnz, mt, optimizationStatistics,
		 computeReferentialValue(B, x, y));
		 printf("Sanity check on l1 constrain %f==%f\n", l1_norm, sqrt(constrain));
		 //----------------- CPU L0 Constrained L1 PCA
		 optimizationSettings->formulation = L0_constrained_L1_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 optimizationSettings, optimizationStatistics);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dasum(y);
		 logTime("L0-Con-L1   ", fval, fval2, nnz, mt, optimizationStatistics,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L1 Constrained L1 PCA
		 optimizationSettings->formulation = L1_constrained_L1_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 optimizationSettings, optimizationStatistics);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dasum(y);
		 logTime("L1-Con-L1   ", fval, fval2, nnz, mt, optimizationStatistics,
		 computeReferentialValue(B, x, y));
		 printf("Sanity check on l1 constrain %f==%f\n", l1_norm, sqrt(constrain));
		 */
		//----------------- Free vectors and matrices
		//	gsl_vector_free(x);
		//		gsl_vector_free(y);
	}
	//	gsl_matrix_free(B);
	//	gsl_matrix_free(BT);
	return 0;
}

int main() {

	sprintf(final_file, "%s%f", filename, gettime());
	fin = fopen(final_file, "w");
//
		test<float> ();
	test<double> ();
init_random_seeds();
	cout << gettime() <<endl;

	fclose(fin);
	return 0;
}


