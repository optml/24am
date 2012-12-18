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
using namespace solver_structures;
#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"
#include "../utils/timer.h"
#include "../problem_generators/gpower_problem_generator.h"
#include "experiment_utils.h"

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
	for (int mult = multSC; mult <= 64; mult = mult * 2) {
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
