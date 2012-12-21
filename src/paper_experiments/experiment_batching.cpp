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
	OptimizationStatistics* optimizationStatistics = new OptimizationStatistics();
	ofstream fileOut;
	fileOut.open("results/paper_experiment_batching.txt");
	mytimer* mt = new mytimer();
	std::vector<F> h_B;
	std::vector<F> x;
	std::vector<F> y;
	int multSC = 1;
	for (int mult = multSC; mult <= 64; mult = mult * 2) {
		int m = 100 * mult;
		int n = 1000 * mult;
		h_B.resize(m * n);
		x.resize(n);
		y.resize(m);
		generateProblem(n, m, &h_B[0], m, n);
		optimizationSettings->maximumIterations = 10;
		optimizationSettings->toll = 0;
		optimizationSettings->totalStartingPoints = 256;
		optimizationSettings->constrain = n / 100;
		optimizationSettings->penalty = 0.02;
		optimizationSettings->formulation = L0_penalized_L2_PCA;
//		optimizationSettings->formulation = L0_constrained_L2_PCA;
		optimizationSettings->onTheFlyMethod = false;
		for (int strategy = 0; strategy < 5; strategy++) {
			switch (strategy) {
			case 0:
				optimizationSettings->batchSize = 1;
				break;
			case 1:
				optimizationSettings->batchSize = 4;
				break;
			case 2:
				optimizationSettings->batchSize = 16;
				break;
			case 3:
				optimizationSettings->batchSize = 64;
				break;
			case 4:
				optimizationSettings->batchSize = optimizationSettings->totalStartingPoints;
				break;
			default:
				break;
			}

			omp_set_num_threads(1);
			init_random_seeds();
			mt->start();
			SPCASolver::MulticoreSolver::denseDataSolver(&h_B[0], m, &x[0], m, n, optimizationSettings,
					optimizationStatistics);
			mt->end();
			logTime(fileOut, mt, optimizationStatistics, optimizationSettings, x, m, n);

			omp_set_num_threads(8);
			init_random_seeds();
			mt->start();
			SPCASolver::MulticoreSolver::denseDataSolver(&h_B[0], m, &x[0], m, n, optimizationSettings,
					optimizationStatistics);
			mt->end();
			logTime(fileOut, mt, optimizationStatistics, optimizationSettings, x, m, n);
		}

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
//template<typename F>
//int test() {
//#ifdef _OPENMP
//	init_random_seeds();
//#endif
//	mytimer* mt = new mytimer();
//	mt->start();
//	std::vector<F> h_B;
//	std::vector<F> x;
//	std::vector<F> y;
//	int multSC = 128;
//	for (int mult = multSC; mult <= 1024; mult = mult * 2) {
//		int m = mult;
//		int n = 10 * mult;
//		h_B.resize(m * n);
//		x.resize(n);
//		y.resize(m);
//		generateProblem(n, m, &h_B[0]);
//
//		optimization_Statisticsistics* optimizationStatistics = new optimization_Statisticsistics();
//
//		optimization_settings* optimizationSettings = new optimization_settings();
//

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
//		optimizationSettings->penalty = 0.01;
//		const F penalty = optimizationSettings->penalty;
//		const unsigned int constrain = optimizationSettings->constrain;
//
//
//
//
//
//			//----------------- CPU L1 Penalized L1 PCA
//			optimizationSettings->formulation = L1_penalized_L1_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1)
//					- penalty * cblas_l1_norm(n, &x[0], 1);
//			logTime("L1-Pen-L1   ", fval, fval2, nnz, mt, optimizationStatistics, optimizationSettings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//			//----------------- CPU L1 Penalized L2 PCA
//			optimizationSettings->formulation = L1_penalized_L2_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l2_norm(m, &y[0], 1)
//					- penalty * cblas_l1_norm(n, &x[0], 1);
//			logTime("L1-Pen-L2 BT", fval, fval2, nnz, mt, optimizationStatistics, optimizationSettings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//----------------- CPU L0 Penalized L1 PCA
//			optimizationSettings->formulation = L0_penalized_L1_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1);
//			fval2 = fval2 * fval2 - penalty * nnz;
//			logTime("L0-Pen-L1   ", fval, fval2, nnz, mt, optimizationStatistics, optimizationSettings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//============= L0 Pen L2
//			optimizationSettings->formulation = L0_penalized_L2_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l2_norm(m, &y[0], 1);
//			fval2 = fval2 * fval2 - penalty * nnz;
//			logTime("L0-Pen-L2 M1", fval, fval2, nnz, mt, optimizationStatistics, optimizationSettings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//----------------- CPU L0 Constrained L2 PCA
//			optimizationSettings->formulation = L0_constrained_L2_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l2_norm(m, &y[0], 1);
//			logTime("L0-Con-L2   ", fval, fval2, nnz, mt, optimizationStatistics, optimizationSettings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//----------------- CPU L1 Constrained L2 PCA
//			optimizationSettings->formulation = L1_constrained_L2_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l2_norm(m, &y[0], 1);
//			l1_norm = cblas_l1_norm(n, &x[0], 1);
//			logTime("L1-Con-L2   ", fval, fval2, nnz, mt, optimizationStatistics, optimizationSettings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//			printf("Sanity check on l1 constrain %f==%f\n", l1_norm,
//					sqrt(constrain));
//			//----------------- CPU L0 Constrained L1 PCA
//			optimizationSettings->formulation = L0_constrained_L1_PCA;
//
//			optimizationSettings->hard_tresholding_using_sort = false;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1);
//			logTime("L0-Con-L1  k", fval, fval2, nnz, mt, optimizationStatistics, optimizationSettings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			optimizationSettings->hard_tresholding_using_sort = true;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1);
//			logTime("L0-Con-L1  s", fval, fval2, nnz, mt, optimizationStatistics, optimizationSettings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//----------------- CPU L1 Constrained L1 PCA
//			optimizationSettings->formulation = L1_constrained_L1_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, optimizationSettings, optimizationStatistics);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1);
//			logTime("L1-Con-L1   ", fval, fval2, nnz, mt, optimizationStatistics, optimizationSettings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//			printf("Sanity check on l1 constrain %f==%f\n", l1_norm,
//					sqrt(constrain));
//
//			//----------------- Free vectors and matrices
//			//	gsl_vector_free(x);
//			//		gsl_vector_free(y);
//
//		}
//	}
//	return 0;
//}

