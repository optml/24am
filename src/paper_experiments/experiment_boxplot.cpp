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

#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
using namespace SolverStructures;
#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "../utils/timer.h"
#include "../problem_generators/gpower_problem_generator.h"
#include "experiment_utils.h"

template<typename F>
void run_experiments(OptimizationSettings* optimizationSettings) {
	OptimizationStatisticsistics* optimizationStatistics = new OptimizationStatisticsistics();
	ofstream fileOut;
	fileOut.open("results/paper_experiment_boxplots.txt");
	mytimer* mt = new mytimer();
	std::vector<F> h_B;
	std::vector<F> x;
	std::vector<F> y;
	int m = 512;
	int n = 2048;
	h_B.resize(m * n);
	x.resize(n);
	y.resize(m);
	generateProblem(n, m, &h_B[0], m, n,true);

	for (optimizationSettings->constrain = 1; optimizationSettings->constrain <= n;
			optimizationSettings->constrain = optimizationSettings->constrain * 2) {
		optimizationSettings->max_it = 20;
		optimizationSettings->toll = 0.000001;
		optimizationSettings->starting_points = 1000;
		optimizationSettings->batch_size = optimizationSettings->starting_points;
		optimizationSettings->algorithm = L0_constrained_L2_PCA;
		optimizationSettings->get_values_for_all_points = true;

		omp_set_num_threads(1);
		init_random_seeds();
		mt->start();
		SPCASolver::MulticoreSolver::denseDataSolver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
		mt->end();
		fileOut << optimizationSettings->constrain << ",";
		for (int i = 0; i < optimizationStatistics->values.size(); i++) {
			fileOut << optimizationStatistics->values[i] << ",";
		}
		fileOut << endl;

	}

	fileOut.close();
}

int main(int argc, char *argv[]) {
	OptimizationSettings* optimizationSettings = new OptimizationSettings();
	run_experiments<double>(optimizationSettings);
	return 0;
}

//

//
//
//
//#include <iostream>
//using namespace std;
//#include <stdio.h>
//#include "gpower/timer.h"
//
//#define GSL_RANGE_CHECK 0
//
//#include <gsl/gsl_cblas.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_vector.h>
//
//#include <gsl/gsl_blas.h>
//
//#include "gpower/openmp_helper.h"
//#include "gpower/helpers.h"
//#include "gpower/optimization_settings.h"
//#include "gpower/optimization_statistics.h"
//#include "gpower/sparse_PCA_solver.h"
//
//unsigned int vector_get_nnz(const gsl_vector * x) {
//	unsigned int nnz = 0;
//#ifdef _OPENMP
//#pragma omp parallel for reduction(+:nnz)
//#endif
//	for (unsigned int i = 0; i < x->size; i++) {
//		if (gsl_vector_get(x, i) != 0)
//			nnz++;
//	}
//	return nnz;
//}
//
//void generate_random_instance(int n, int m, gsl_matrix * B, gsl_matrix * BT,
//		gsl_vector * x) {
//
//#ifdef _OPENMP
//	//#pragma omp parallel for
//#endif
//	for (int i = 0; i < n; i++) {
//		gsl_vector_set(x, i, (double) rand_r(&myseed) / RAND_MAX);
//		for (int j = 0; j < m; j++) {
//			double tmp = (double) rand_r(&myseed) / RAND_MAX;
//			tmp = tmp * 2 - 1;
//			gsl_matrix_set(B, j, i, tmp);
//		}
//		gsl_vector_view col = gsl_matrix_column(B, i);
//		double col_norm = gsl_blas_dnrm2(&col.vector);
//		gsl_vector_scale(&col.vector, 1 / col_norm);
//	}
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
//	for (int i = 0; i < n; i++) {
//		for (int j = 0; j < m; j++) {
//			gsl_matrix_set(BT, i, j, gsl_matrix_get(B, j, i));
//		}
//	}
//}
//
//void getFileSize(const char* filename, int& DIM_M, int& DIM_N) {
//	FILE * fin = fopen(filename, "r");
//	if (fin == NULL) {
//
//	} else {
//		fscanf(fin, "%d;%d", &DIM_M, &DIM_N);
//		fclose(fin);
//	}
//}
//
//void readFromFile(const char* filename, int& DIM_M, int& DIM_N, gsl_matrix * B,
//		gsl_matrix * BT) {
//	int i, j;
//	FILE * fin = fopen(filename, "r");
//	if (fin == NULL) {
//	} else {
//		fscanf(fin, "%d;%d", &DIM_M, &DIM_N);
//		for (j = 0; j < DIM_M; j++) {
//			for (i = 0; i < DIM_N; i++) {
//				float tmp = -1;
//				fscanf(fin, "%f;", &tmp);
//				gsl_matrix_set(B, j, i, tmp);
//				gsl_matrix_set(BT, i, j, tmp);
//			}
//		}
//		fclose(fin);
//	}
//}
//
//void logTime(const char* label, double fval, double fval2, int nnz,
//		mytimer* mt, optimization_Statisticsistics* optimizationStatistics, double refval) {
//	printf("%s,%1.5f,%1.5f,%d,%f,%f,%d,%f\n", label, fval, fval2, nnz,
//			mt->getElapsedCPUTime(), mt->getElapsedWallClockTime(), optimizationStatistics->it,
//			refval);
//}
//
//double computeReferentialValue(gsl_matrix * B, gsl_vector * x, gsl_vector * y) {
//	gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
//	double fval2 = gsl_blas_dnrm2(y);
//	return fval2 * fval2;
//}
//
//template<typename F>
//int generateProblem(const int n, const int m, F* h_B) {
//
//	printf("%d x %d \n", m, n);
//
//	char final_file[1000];
//	unsigned int seed = 0;
//	for (int i = 0; i < n; i++) {
//		double norm = 0;
//		for (int j = 0; j < m; j++) {
//			F tmp = (F) rand_r(&seed) / RAND_MAX;
//			h_B[j + i * m] = -1 + 2 * tmp;
//			norm += h_B[j + i * m] * h_B[j + i * m];
//		}
//		norm = 1 / sqrt(norm);
//		norm=norm*(rand()/(0.0+RAND_MAX));
//		for (int j = 0; j < m; j++) {
//			h_B[j + i * m] = h_B[j + i * m] * norm;
//		}
//
//	}
//	return 0;
//}
//
//template<typename F>
//int test() {
//
//#ifdef _OPENMP
//	init_random_seeds();
//#endif
//
//	mytimer* mt = new mytimer();
//	mt->start();
//
//	//	for ( mult <= 100; mult = mult * 2)
//
//	int mult = 50;
//	{
//		int m = 20 * mult;
//		int n = 30 * mult;
//		n = 2048;
//		m = 512;
//		gsl_matrix * B;
//		gsl_matrix * BT;
//		//	gsl_vector * x;
//		F h_B[m * n];
//		F x[n];
//		if (true) {
//			generateProblem(n, m, h_B);
//			//		B = gsl_matrix_alloc(m, n);
//			//		BT = gsl_matrix_alloc(n, m);
//			//		x = gsl_vector_alloc(n);
//			//		generate_random_instance(n, m, B, BT, x);
//		} else {
//			getFileSize("/document/phd/c/GPower/resources/data.txt0", m, n);
//			B = gsl_matrix_alloc(m, n);
//			BT = gsl_matrix_alloc(n, m);
//			readFromFile("/document/phd/c/GPower/resources/data.txt0", m, n, B,
//					BT);
//
//			//		x = gsl_vector_alloc(n);
//		}
//
//		gsl_vector * y = gsl_vector_alloc(m);
//
//		optimization_Statisticsistics* optimizationStatistics = new optimization_Statisticsistics();
//
//		optimization_settings* optimizationSettings = new optimization_settings();
//
//		optimizationSettings->max_it = 200;
//		optimizationSettings->toll = 0.000001;
//		optimizationSettings->starting_points = 1000;
//				optimizationSettings->batch_size=optimizationSettings->starting_points;
//
//		F fval = 0;
//		F fval2 = 0;
//		F l1_norm;
//		unsigned int nnz = 0;
//		mt->end();
//		cout << "Problem generation took " << mt->getElapsedCPUTime() << " "
//				<< mt->getElapsedWallClockTime() << endl;
//		//============================
//		optimizationSettings->constrain = 10;
//		optimizationSettings->penalty = 0.00001;
//		const F penalty = optimizationSettings->penalty;
//		const unsigned int constrain = optimizationSettings->constrain;
//
//		SparsePCA_Algorithm algorithms[8];
//
//		algorithms[0] = L0_penalized_L1_PCA;
//		algorithms[1] = L0_penalized_L2_PCA;
//		algorithms[2] = L1_penalized_L1_PCA;
//		algorithms[3] = L1_penalized_L2_PCA;
//		algorithms[4] = L0_constrained_L1_PCA;
//		algorithms[5] = L0_constrained_L2_PCA;
//		algorithms[6] = L1_constrained_L1_PCA;
//		algorithms[7] = L1_constrained_L2_PCA;
//
//
//
//		optimizationSettings->penalty = 0;
//		optimizationSettings->constrain = 0;
//
////		FILE * fin =
////				fopen("/document/phd/c/GPower/resources/boxplots.txt", "w");
////		for (optimizationSettings->penalty = 2*  4.096; optimizationSettings->penalty >= 0.00001; optimizationSettings->penalty
////				= optimizationSettings->penalty * 0.65) {
////		for (int alg =1; alg < 2; alg++) {
//
//
//				FILE * fin = fopen("/document/phd/c/GPower/resources/boxplots_con.txt",
//						"w");
//					for (optimizationSettings->constrain = 1; optimizationSettings->constrain <= n; optimizationSettings->constrain
//							= optimizationSettings->constrain * 2) {
//
//
//
//			for (int alg =5; alg < 6; alg++) {
//
//
//				optimizationSettings->algorithm = algorithms[alg];
//				mt->start();
//
//
//
//
//				optimizationStatistics->fval = sparse_PCA_solver(h_B, m, x, m, n, optimizationSettings, optimizationStatistics);
//
//
//
//				int nnz = vector_get_nnz(x, n);
//				printf("%f,%d,%d,%d,%d,%f,%d\n", optimizationStatistics->fval, optimizationStatistics->it, alg, m,
//						n, optimizationSettings->constrain + 0.0 + optimizationSettings->penalty, nnz);
//				fprintf(fin, "%f,%d,%d,%d,%d,%f,%d", optimizationStatistics->fval, optimizationStatistics->it, alg,
//						m, n, optimizationSettings->constrain + 0.0 + optimizationSettings->penalty,
//						nnz);
//				for (int i = 0; i < optimizationSettings->starting_points; i++) {
//					fprintf(fin, ",%f", optimizationStatistics->values[i]);
//				}
//				for (int i = 0; i < optimizationSettings->starting_points; i++) {
//					fprintf(fin, ",%d", optimizationStatistics->cardinalities[i]);
//				}
//				fprintf(fin, "\n");
//
//				mt->end();
//
//			}
//		}
//		fclose(fin);
//
//		gsl_vector_free(y);
//		gsl_matrix_free(B);
//		gsl_matrix_free(BT);
//	}
//
//	/*	//----------------- CPU L1 Penalized L1 PCA
//	 optimizationSettings->algorithm = L1_penalized_L1_PCA;
//	 mt->start();
//	 fval = sparse_PCA_solver(h_B, m, x->data, B->size1, B->size2,
//	 optimizationSettings, optimizationStatistics);
//	 mt->end();
//	 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
//	 nnz = vector_get_nnz(x);
//	 fval2 = gsl_blas_dasum(y) - penalty * gsl_blas_dasum(x);
//	 logTime("L1-Pen-L1   ", fval, fval2, nnz, mt, optimizationStatistics,
//	 computeReferentialValue(B, x, y));
//
//	 //----------------- CPU L1 Penalized L2 PCA
//	 optimizationSettings->algorithm = L1_penalized_L2_PCA;
//	 mt->start();
//	 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
//	 optimizationSettings, optimizationStatistics);
//	 mt->end();
//	 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
//	 nnz = vector_get_nnz(x);
//	 fval2 = gsl_blas_dnrm2(y) - penalty * gsl_blas_dasum(x);
//	 logTime("L1-Pen-L2 BT", fval, fval2, nnz, mt, optimizationStatistics,
//	 computeReferentialValue(B, x, y));
//
//	 //----------------- CPU L0 Penalized L1 PCA
//	 optimizationSettings->algorithm = L0_penalized_L1_PCA;
//	 mt->start();
//	 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
//	 optimizationSettings, optimizationStatistics);
//	 mt->end();
//	 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
//	 nnz = vector_get_nnz(x);
//	 fval2 = gsl_blas_dasum(y);
//	 fval2 = fval2 * fval2 - penalty * nnz;
//	 logTime("L0-Pen-L1   ", fval, fval2, nnz, mt, optimizationStatistics,
//	 computeReferentialValue(B, x, y));
//
//	 //============= L0 Pen L2
//	 optimizationSettings->algorithm = L0_penalized_L2_PCA;
//	 mt->start();
//	 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
//	 optimizationSettings, optimizationStatistics);
//	 mt->end();
//	 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
//	 nnz = vector_get_nnz(x);
//	 fval2 = gsl_blas_dnrm2(y);
//	 fval2 = fval2 * fval2 - penalty * nnz;
//	 logTime("L0-Pen-L2 M1", fval, fval2, nnz, mt, optimizationStatistics,
//	 computeReferentialValue(B, x, y));
//
//	 //----------------- CPU L0 Constrained L2 PCA
//	 optimizationSettings->algorithm = L0_constrained_L2_PCA;
//	 mt->start();
//	 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
//	 optimizationSettings, optimizationStatistics);
//	 mt->end();
//	 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
//	 nnz = vector_get_nnz(x);
//	 fval2 = gsl_blas_dnrm2(y);
//	 logTime("L0-Con-L2   ", fval, fval2, nnz, mt, optimizationStatistics,
//	 computeReferentialValue(B, x, y));
//
//	 //----------------- CPU L1 Constrained L2 PCA
//	 optimizationSettings->algorithm = L1_constrained_L2_PCA;
//	 mt->start();
//	 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
//	 optimizationSettings, optimizationStatistics);
//	 mt->end();
//	 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
//	 nnz = vector_get_nnz(x);
//	 fval2 = gsl_blas_dnrm2(y);
//	 l1_norm = gsl_blas_dasum(x);
//	 logTime("L1-Con-L2   ", fval, fval2, nnz, mt, optimizationStatistics,
//	 computeReferentialValue(B, x, y));
//	 printf("Sanity check on l1 constrain %f==%f\n", l1_norm, sqrt(constrain));
//	 //----------------- CPU L0 Constrained L1 PCA
//	 optimizationSettings->algorithm = L0_constrained_L1_PCA;
//	 mt->start();
//	 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
//	 optimizationSettings, optimizationStatistics);
//	 mt->end();
//	 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
//	 nnz = vector_get_nnz(x);
//	 fval2 = gsl_blas_dasum(y);
//	 logTime("L0-Con-L1   ", fval, fval2, nnz, mt, optimizationStatistics,
//	 computeReferentialValue(B, x, y));
//
//	 //----------------- CPU L1 Constrained L1 PCA
//	 optimizationSettings->algorithm = L1_constrained_L1_PCA;
//	 mt->start();
//	 fval = sparse_PCA_solver(BT->data, BT->tda, x->data, B->size1, B->size2,
//	 optimizationSettings, optimizationStatistics);
//	 mt->end();
//	 gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
//	 nnz = vector_get_nnz(x);
//	 fval2 = gsl_blas_dasum(y);
//	 logTime("L1-Con-L1   ", fval, fval2, nnz, mt, optimizationStatistics,
//	 computeReferentialValue(B, x, y));
//	 printf("Sanity check on l1 constrain %f==%f\n", l1_norm, sqrt(constrain));
//	 */
//	//----------------- Free vectors and matrices
//	//	gsl_vector_free(x);
//
//	return 0;
//}
//
//int main() {
//	test<double> ();
//	//	test<double> ();
//}
