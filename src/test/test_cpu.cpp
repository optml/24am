//============================================================================
// Name        : personal_pc.cpp
// Author      : Martin Takac
// Version     :
// Copyright   : GNU
// Description : GPower Method
//============================================================================
#include <iostream>
using namespace std;
#include <stdio.h>
#include "../utils/timer.h"
#include "../utils/gsl_helper.h"
#include "../utils/openmp_helper.h"
#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
using namespace solver_structures ;
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
		mytimer* mt, optimization_statistics* stat, double refval) {
	printf("%s,%1.5f,%1.5f,%d,%f,%f,%d,%f\n", label, fval, fval2, nnz,
			mt->getElapsedCPUTime(), mt->getElapsedWallClockTime(), stat->it,
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

		optimization_statistics* stat = new optimization_statistics();

		optimization_settings* settings = new optimization_settings();

		settings->max_it = 10;
		settings->toll = 0.;
		settings->starting_points = 1;

		F fval = 0;
		F fval2 = 0;
		F l1_norm;
		unsigned int nnz = 0;
		mt->end();
		cout << "Problem generation took " << mt->getElapsedCPUTime() << " "
				<< mt->getElapsedWallClockTime() << endl;
		//============================
		settings->constrain = 10;
		settings->penalty = 0.01;
		const F penalty = settings->penalty;
		const unsigned int constrain = settings->constrain;

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
			settings->algorithm = algorithms[alg];
			for (settings->starting_points = 1; settings->starting_points
					<= 128 * 8*4; //
			settings->starting_points = settings->starting_points * 2+1) {
				mt->start();
				stat->fval = PCA_solver::dense_PCA_solver(&h_B[0], m, &x[0], m, n,
						settings, stat);
				mt->end();
				printf(
						"%d,%d,%f,%f,%d,%d,%d,%d\n",
						alg,
						settings->starting_points,
						stat->true_computation_time,
						stat->true_computation_time / (0.0
								+ settings->starting_points * settings->max_it),
						m, n, TOTAL_THREADS, sizeof(F));
				fprintf(
						fin,
						"%d,%d,%f,%f,%d,%d,%d,%d\n",
						alg,
						settings->starting_points,
						stat->true_computation_time,
						stat->true_computation_time / (0.0
								+ settings->starting_points * settings->max_it),
						m, n, TOTAL_THREADS, sizeof(F));

			}
		}
		/*	//----------------- CPU L1 Penalized L1 PCA
		 settings->algorithm = L1_penalized_L1_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(h_B, m, x->data, B->size1, B->size2,
		 settings, stat);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dasum(y) - penalty * gsl_blas_dasum(x);
		 logTime("L1-Pen-L1   ", fval, fval2, nnz, mt, stat,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L1 Penalized L2 PCA
		 settings->algorithm = L1_penalized_L2_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 settings, stat);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dnrm2(y) - penalty * gsl_blas_dasum(x);
		 logTime("L1-Pen-L2 BT", fval, fval2, nnz, mt, stat,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L0 Penalized L1 PCA
		 settings->algorithm = L0_penalized_L1_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 settings, stat);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dasum(y);
		 fval2 = fval2 * fval2 - penalty * nnz;
		 logTime("L0-Pen-L1   ", fval, fval2, nnz, mt, stat,
		 computeReferentialValue(B, x, y));

		 //============= L0 Pen L2
		 settings->algorithm = L0_penalized_L2_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 settings, stat);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dnrm2(y);
		 fval2 = fval2 * fval2 - penalty * nnz;
		 logTime("L0-Pen-L2 M1", fval, fval2, nnz, mt, stat,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L0 Constrained L2 PCA
		 settings->algorithm = L0_constrained_L2_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 settings, stat);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dnrm2(y);
		 logTime("L0-Con-L2   ", fval, fval2, nnz, mt, stat,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L1 Constrained L2 PCA
		 settings->algorithm = L1_constrained_L2_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 settings, stat);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dnrm2(y);
		 l1_norm = gsl_blas_dasum(x);
		 logTime("L1-Con-L2   ", fval, fval2, nnz, mt, stat,
		 computeReferentialValue(B, x, y));
		 printf("Sanity check on l1 constrain %f==%f\n", l1_norm, sqrt(constrain));
		 //----------------- CPU L0 Constrained L1 PCA
		 settings->algorithm = L0_constrained_L1_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 settings, stat);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dasum(y);
		 logTime("L0-Con-L1   ", fval, fval2, nnz, mt, stat,
		 computeReferentialValue(B, x, y));

		 //----------------- CPU L1 Constrained L1 PCA
		 settings->algorithm = L1_constrained_L1_PCA;
		 mt->start();
		 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
		 settings, stat);
		 mt->end();
		 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
		 nnz = vector_get_nnz(x);
		 fval2 = gsl_blas_dasum(y);
		 logTime("L1-Con-L1   ", fval, fval2, nnz, mt, stat,
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


