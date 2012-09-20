/*
 //HEADER INFO
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
//		load_data_and_run_solver<float>(settings);
	}
	MPI_Finalize();
	return 0;
}

/*==== MAIN FUNCTION =================================================*/
int mainX(int argc, char *argv[]) {

//
//	int XGRID = 1;
//
//	double start_time = gettime();
//	int node = 0;
////	node = distributed_pca_solver(file, outputfile, files, files, &settings, &stat);
//	node = distributed_pca_solver_from_two_dim_files(XGRID, &settings, &stat);
//	double end_time = gettime();
//	stat.total_elapsed_time = end_time - start_time;
//	if (node == 0) {
//		save_statistics_and_results(&settings, &stat);
//	}

	return 0;
}

