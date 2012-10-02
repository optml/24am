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
 *    DISTRIBUTIVE SOLVER FOR SPARSE PCA
 *
 * 
 */

/*
 [1 2
 3 4]
 is stored as  1 3 2 4!!!
 */

/* Header files*/
#include <stdio.h>
#include <stdlib.h>
unsigned int myseed = 0;

#include "../dgpower/distributed_PCA_solver.h"

#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"

template<typename F>
void run_solver(solver_structures::optimization_settings * settings) {
	solver_structures::optimization_statistics* stat =
			new optimization_statistics();
	MKL_INT iam, nprocs;
	blacs_pinfo_(&iam, &nprocs);
	double start_all = gettime();
	PCA_solver::distributed_solver::optimization_data<F> optimization_data_inst;
	PCA_solver::distributed_solver::load_data_from_2d_files_and_distribution<F>(
			optimization_data_inst, settings, stat);

	/*
	 *  RUN SOLVER
	 */
	double start_time = gettime();
	PCA_solver::distributed_solver::distributed_sparse_PCA_solver(
			optimization_data_inst, settings, stat);
	double end_time = gettime();
	stat->true_computation_time = end_time - start_time;
	/*
	 * STORE RESULT INTO FILE
	 */
	PCA_solver::distributed_solver::gather_and_store_best_result_to_file(
			optimization_data_inst, settings, stat);
	if (iam == 0) {
		stat->total_elapsed_time = gettime() - start_all;
		input_ouput_helper::save_statistics(stat, settings);
	}

	blacs_gridexit_(&optimization_data_inst.params.ictxt);
}


int main(int argc, char *argv[]) {
	solver_structures::optimization_settings* settings =
			new optimization_settings();
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &settings->proccess_node);
	int status = parse_console_options(settings, argc, argv);
	if (status > 0) {
		MPI_Finalize();
		return status;
	}
	if (settings->double_precission) {
		run_solver<double>(settings);
	} else {
		run_solver<float>(settings);
	}
	MPI_Finalize();
	return 0;
}

