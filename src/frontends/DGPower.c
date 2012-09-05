/*
 *    DISTRIBUTIVE SOLVER FOR SPARSE PCA
 *
 *   todo
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

#include "dgpower/time_helper.h"
#include "dgpower/optimization_settings.h"
#include "dgpower/optimization_statistics.h"
#include "dgpower/distributed_pca_solver.h"


#ifndef MAXIT
#define MAXIT 1
#endif


#ifndef FILES
#define FILES 720
#endif

/*==== MAIN FUNCTION =================================================*/
int main(int argc, char *argv[]) {

	//	char* file = "/document/phd/c/GPower/resources/pca.dat";
//		char* file = "/document/phd/c/GPower/resources/known.txt";
//		char* outputfile = "/document/phd/c/GPower/resources/output.dat";

//	char* file = "/exports/work/maths_oro/taki/generated.dat.";
	char* outputfile = "/exports/work/maths_oro/taki/output.dat";

	char* file = "/exports/work/scratch/taki/generated.dat.";



	int XGRID = 10;
	struct optimization_settings settings;
	struct optimization_statistics stat;
	settings.toll = 0.00;

	settings.max_it = MAXIT;
	settings.penalty = 0.0001;
	settings.starting_points = 1;

	settings.algorithm = L0_penalized_L1_PCA;
	settings.algorithm = L1_penalized_L1_PCA;
	settings.algorithm = L0_penalized_L2_PCA;
	settings.algorithm = L1_penalized_L2_PCA;

	double start_time = gettime();
	int node = 0;
//	node = distributed_pca_solver(file, outputfile, files, files, &settings, &stat);
	node = distributed_pca_solver_from_two_dim_files(file, outputfile, XGRID,   &settings, &stat);
	double end_time = gettime();
	if (node == 0) {
		printf("ALG-with total solving time;%f;%d;%f\n", end_time - start_time, stat.it, stat.fval);
	}




	return 0;
}

