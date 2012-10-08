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
using namespace solver_structures;
#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "../utils/timer.h"
#include "../problem_generators/gpower_problem_generator.h"

template<typename F>
void logTime(ofstream &stream, mytimer* mt, optimization_statistics* stat,
		optimization_settings* settings, std::vector<F>& x, int m, int n) {
	int nnz = vector_get_nnz(&x[0], n);
	cout << settings->algorithm << "," << nnz << "," << m << "," << n << ","
			<< mt->getElapsedWallClockTime() << ","
			<< stat->true_computation_time << "," << settings->batch_size << ","
			<< settings->on_the_fly_generation
			<< ","<<stat->total_threads_used
			<< ","<<settings->starting_points
			<< ","<<stat->it
			<< endl;
	stream<< settings->algorithm << "," << nnz << "," << m << "," << n << ","
			<< mt->getElapsedWallClockTime() << ","
			<< stat->true_computation_time << "," << settings->batch_size << ","
			<< settings->on_the_fly_generation
			<< ","<<stat->total_threads_used
			<< ","<<settings->starting_points
			<< ","<<stat->it
			<< endl;
}

template<typename F>
void run_experiments(optimization_settings* settings) {
	optimization_statistics* stat = new optimization_statistics();
	ofstream fileOut;
	fileOut.open("results/paper_experiment_otf.txt");
	mytimer* mt = new mytimer();
	std::vector<F> h_B;
	std::vector<F> x;
	std::vector<F> y;
	int multSC = 1;
	for (int mult = multSC; mult <= 128; mult = mult * 2) {
		int m = 100*mult;
		int n = 1000 * mult;
		h_B.resize(m * n);
		x.resize(n);
		y.resize(m);
		generateProblem(n, m, &h_B[0], m, n);
		settings->max_it = 100;
		settings->toll = 0.01;
		settings->starting_points = 1024;
		settings->penalty=0.02;
		settings->constrain=n/100;
		settings->algorithm=L0_penalized_L2_PCA;
		for (int strategy = 0; strategy < 3; strategy++) {
			switch (strategy) {
			case 0:
				settings->batch_size = 64;
				settings->on_the_fly_generation = false;
				break;
			case 1:
				settings->batch_size = 64;
				settings->on_the_fly_generation = true;
				break;
			case 2:
				settings->batch_size = settings->starting_points;
				settings->on_the_fly_generation = false;
				break;
			default:
				break;
			}

			omp_set_num_threads(1);
			init_random_seeds();
			mt->start();
			PCA_solver::dense_PCA_solver(&h_B[0], m, &x[0], m, n, settings,
					stat);
			mt->end();
			logTime(fileOut, mt, stat, settings, x, m, n);


		}

	}

	fileOut.close();
}

int main(int argc, char *argv[]) {
	optimization_settings* settings = new optimization_settings();
	run_experiments<double>(settings);
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
//		optimization_statistics* stat = new optimization_statistics();
//
//		optimization_settings* settings = new optimization_settings();
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
//		settings->constrain = 10;
//		settings->penalty = 0.01;
//		const F penalty = settings->penalty;
//		const unsigned int constrain = settings->constrain;
//
//
//
//
//
//			//----------------- CPU L1 Penalized L1 PCA
//			settings->algorithm = L1_penalized_L1_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, settings, stat);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1)
//					- penalty * cblas_l1_norm(n, &x[0], 1);
//			logTime("L1-Pen-L1   ", fval, fval2, nnz, mt, stat, settings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//			//----------------- CPU L1 Penalized L2 PCA
//			settings->algorithm = L1_penalized_L2_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, settings, stat);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l2_norm(m, &y[0], 1)
//					- penalty * cblas_l1_norm(n, &x[0], 1);
//			logTime("L1-Pen-L2 BT", fval, fval2, nnz, mt, stat, settings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//----------------- CPU L0 Penalized L1 PCA
//			settings->algorithm = L0_penalized_L1_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, settings, stat);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1);
//			fval2 = fval2 * fval2 - penalty * nnz;
//			logTime("L0-Pen-L1   ", fval, fval2, nnz, mt, stat, settings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//============= L0 Pen L2
//			settings->algorithm = L0_penalized_L2_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, settings, stat);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l2_norm(m, &y[0], 1);
//			fval2 = fval2 * fval2 - penalty * nnz;
//			logTime("L0-Pen-L2 M1", fval, fval2, nnz, mt, stat, settings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//----------------- CPU L0 Constrained L2 PCA
//			settings->algorithm = L0_constrained_L2_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, settings, stat);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l2_norm(m, &y[0], 1);
//			logTime("L0-Con-L2   ", fval, fval2, nnz, mt, stat, settings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//----------------- CPU L1 Constrained L2 PCA
//			settings->algorithm = L1_constrained_L2_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, settings, stat);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l2_norm(m, &y[0], 1);
//			l1_norm = cblas_l1_norm(n, &x[0], 1);
//			logTime("L1-Con-L2   ", fval, fval2, nnz, mt, stat, settings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//			printf("Sanity check on l1 constrain %f==%f\n", l1_norm,
//					sqrt(constrain));
//			//----------------- CPU L0 Constrained L1 PCA
//			settings->algorithm = L0_constrained_L1_PCA;
//
//			settings->hard_tresholding_using_sort = false;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, settings, stat);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1);
//			logTime("L0-Con-L1  k", fval, fval2, nnz, mt, stat, settings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			settings->hard_tresholding_using_sort = true;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, settings, stat);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1);
//			logTime("L0-Con-L1  s", fval, fval2, nnz, mt, stat, settings, m, n,
//					computeReferentialValue(&h_B[0], &x[0], &y[0], m, n));
//
//			//----------------- CPU L1 Constrained L1 PCA
//			settings->algorithm = L1_constrained_L1_PCA;
//			mt->start();
//			fval = sparse_PCA_solver(&h_B[0], m, &x[0], m, n, settings, stat);
//			mt->end();
//			cblas_matrix_matrix_multiply(CblasColMajor, CblasNoTrans,
//					CblasNoTrans, m, 1, n, 1, &h_B[0], m, &x[0], n, 0, &y[0],
//					m);
//			nnz = vector_get_nnz(&x[0], n);
//			fval2 = cblas_l1_norm(m, &y[0], 1);
//			logTime("L1-Con-L1   ", fval, fval2, nnz, mt, stat, settings, m, n,
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

