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

/*
 * option_parser.h
 *
 *  Created on: Sep 12, 2012
 *      Author: taki
 */

#ifndef OPTION_PARSER_H_
#define OPTION_PARSER_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../class/optimization_settings.h"
using namespace std;
using namespace SolverStructures;

void print_usage(){
	cout << "Usage:"<<endl;
	cout << "-------------------------------------"<<endl;
	cout << "Required Parameters:"<<endl;

	cout << "-------------------------------------"<<endl;
	cout << "Optional Parameters:"<<endl;

}

int parseConsoleOptions(SolverStructures::OptimizationSettings* optimizationSettings,
		int argc, char *argv[]) {

	char c;
	/*
	 * d - data file
	 * r - result file
	 * i - max number of iterations (*optional*)
	 * t - tolerance (*optional*)
	 * s - number of starting points (*optional*)
	 * b - batch sizes (*optional*)
	 * u - batching type (*optional*)
	 * v - verbose (*optional*) default false
	 * p - use DOUBLE precission (*optional*)
	 * a - algorithm
	 * n - constrain parameter
	 * m - penalty parameter
	 * x - x-dimension of distributed files (FOR DISTRIBUTED METHOD ONLY)
	 */
	bool data_file = false;
	bool result_file = false;
	bool algorithm = false;
	while ((c = getopt(argc, argv, "d:f:r:i:t:s:b:u:v:p:a:n:m:x:")) != -1) {
		switch (c) {
		case 'x':
			optimizationSettings->distributed_row_grid_file = atoi(optarg);
			break;
		case 's':
			optimizationSettings->starting_points = atoi(optarg);
			break;
		case 'b':
			optimizationSettings->batch_size = atoi(optarg);
			break;
		case 'n':
			optimizationSettings->constrain = atoi(optarg);
			break;
		case 'm':
			optimizationSettings->penalty= atof(optarg);
			break;
		case 'i':
			optimizationSettings->max_it = atoi(optarg);
			break;
		case 'd':
			optimizationSettings->data_file = optarg;
			data_file = true;
			break;
		case 'r':
			optimizationSettings->result_file = optarg;
			result_file = true;
			break;
		case 'u':
			optimizationSettings->onTheFlyMethod= atoi(optarg);
			break;
		case 't':
			optimizationSettings->toll = atof(optarg);
			break;
		case 'p':
			optimizationSettings->double_precission = true;
			break;
		case 'v':
			optimizationSettings->verbose = true;
			break;
		case 'a':
			switch (atoi(optarg)) {
			case 1:
				optimizationSettings->algorithm=L0_constrained_L1_PCA;
				algorithm=true;
				break;
			case 2:
				optimizationSettings->algorithm=L0_constrained_L2_PCA;
				algorithm=true;
				break;
			case 3:
				optimizationSettings->algorithm=L1_constrained_L1_PCA;
				algorithm=true;
				break;
			case 4:
				optimizationSettings->algorithm=L1_constrained_L2_PCA;
				algorithm=true;
				break;
			case 5:
				optimizationSettings->algorithm=L0_penalized_L1_PCA;
				algorithm=true;
				break;
			case 6:
				optimizationSettings->algorithm=L0_penalized_L2_PCA;
				algorithm=true;
				break;
			case 7:
				optimizationSettings->algorithm=L1_penalized_L1_PCA;
				algorithm=true;
				break;
			case 8:
				optimizationSettings->algorithm=L1_penalized_L2_PCA;
				algorithm=true;
				break;

			}

			break;
		}
	}

	if (!data_file || !result_file || !algorithm) {
		if (optimizationSettings->proccess_node==0)
			print_usage();
		return 1;
	}

	if (optimizationSettings->batch_size > optimizationSettings->starting_points) {
		optimizationSettings->starting_points = optimizationSettings->batch_size;
	}
	return 0;
}

#endif /* OPTION_PARSER_H_ */
