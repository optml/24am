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
void runSolver(SolverStructures::OptimizationSettings * optimizationSettings) {
	SolverStructures::OptimizationStatistics* optimizationStatistics =
			new OptimizationStatistics();
	MKL_INT iam, nprocs;
	blacs_pinfo_(&iam, &nprocs);
	double start_all = gettime();
	SPCASolver::DistributedClasses::OptimizationData<F> optimizationDataInstance;
	SPCASolver::DistributedSolver::loadDataFrom2DFilesAndDistribute<F>(
			optimizationDataInstance, optimizationSettings, optimizationStatistics);

	/*
	 *  RUN SOLVER
	 */
	double start_time = gettime();
	SPCASolver::DistributedSolver::denseDataSolver(
			optimizationDataInstance, optimizationSettings, optimizationStatistics);
	double end_time = gettime();
	optimizationStatistics->totalTrueComputationTime = end_time - start_time;
	/*
	 * STORE RESULT INTO FILE
	 */
	SPCASolver::DistributedSolver::gatherAndStoreBestResultToOutputFile(
			optimizationDataInstance, optimizationSettings, optimizationStatistics);
	if (iam == 0) {
		optimizationStatistics->totalElapsedTime = gettime() - start_all;
		InputOuputHelper::saveSolverStatistics(optimizationStatistics, optimizationSettings);
	}

	blacs_gridexit_(&optimizationDataInstance.params.ictxt);
}


int main(int argc, char *argv[]) {
	SolverStructures::OptimizationSettings* optimizationSettings =
			new OptimizationSettings();
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &optimizationSettings->proccessNode);
	int optimizationStatisticsus = parseConsoleOptions(optimizationSettings, argc, argv);
	if (optimizationStatisticsus > 0) {
		MPI_Finalize();
		return optimizationStatisticsus;
	}
	if (optimizationSettings->double_precission) {
		runSolver<double>(optimizationSettings);
	} else {
		runSolver<float>(optimizationSettings);
	}
	MPI_Finalize();
	return 0;
}

