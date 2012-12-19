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
 *    MULTICORE SOLVER FOR SPARSE PCA - frontend console interface
 *
 */

#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
using namespace SolverStructures;
#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"

template<typename F>
void load_data_and_run_solver(OptimizationSettings* optimizationSettings) {
	double start_wall_time = gettime();
	std::vector<F> B_mat;
	unsigned int ldB, m, n;
	// load data from CSV file
	InputOuputHelper::readCSVFile(B_mat, ldB, m, n, optimizationSettings->dataFilePath);
	OptimizationStatistics* optimizationStatistics = new OptimizationStatistics();
	optimizationStatistics->n = n;
	std::vector<F> x_vec(n, 0);
	// run SOLVER
	SPCASolver::MulticoreSolver::denseDataSolver(&B_mat[0], ldB, &x_vec[0], m, n, optimizationSettings,
			optimizationStatistics);
	double end_wall_time = gettime();
	optimizationStatistics->totalElapsedTime = end_wall_time - start_wall_time;
    // store result into file
	InputOuputHelper::save_results(optimizationStatistics, optimizationSettings, &x_vec[0], n);
	// store OptimizationStatistics into optimizationStatistics file
	InputOuputHelper::saveSolverStatistics(optimizationStatistics, optimizationSettings);
}

int main(int argc, char *argv[]) {
	OptimizationSettings* optimizationSettings = new OptimizationSettings();
	int optimizationStatisticsus = parseConsoleOptions(optimizationSettings, argc, argv);
	if (optimizationStatisticsus > 0)
		return optimizationStatisticsus;
	if (optimizationSettings->double_precission) {
		load_data_and_run_solver<double>(optimizationSettings);
	} else {
		load_data_and_run_solver<float>(optimizationSettings);
	}
	return 0;
}
