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
 *    DISTRIBUTIVE SOLVER FOR SPARSE PCA - frontend console interface
 *
 * 
 */

/*
 Data-matrix is stored column-wise
 */

/* Header files*/
#include <stdio.h>
#include <stdlib.h>
#include "../dgpower/distributed_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"

template<typename F>
void runSolver(SolverStructures::OptimizationSettings * settings) {
	SolverStructures::OptimizationStatistics* stat =
			new OptimizationStatistics();
	MKL_INT iam, nprocs;
	blacs_pinfo_(&iam, &nprocs);
	double start_all = gettime();
	SPCASolver::DistributedClasses::OptimizationData<F> optimization_data_inst;
	SPCASolver::DistributedSolver::loadDataFrom2DFilesAndDistribute<F>(
			optimization_data_inst, settings, stat);

	/*
	 *  RUN SOLVER
	 */
	double start_time = gettime();
	SPCASolver::DistributedSolver::denseDataSolver(
			optimization_data_inst, settings, stat);
	double end_time = gettime();
	stat->true_computation_time = end_time - start_time;
	/*
	 * STORE RESULT INTO FILE
	 */
	SPCASolver::DistributedSolver::gather_and_store_best_result_to_file(
			optimization_data_inst, settings, stat);
	if (iam == 0) {
		stat->total_elapsed_time = gettime() - start_all;
		InputOuputHelper::save_statistics(stat, settings);
	}

	blacs_gridexit_(&optimization_data_inst.params.ictxt);
}


int main(int argc, char *argv[]) {
	SolverStructures::OptimizationSettings* settings =
			new OptimizationSettings();
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &settings->proccess_node);
	int status = parseConsoleOptions(settings, argc, argv);
	if (status > 0) {
		MPI_Finalize();
		return status;
	}
	if (settings->double_precission) {
		runSolver<double>(settings);
	} else {
		runSolver<float>(settings);
	}
	MPI_Finalize();
	return 0;
}

