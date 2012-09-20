/*
//HEADER INFO
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
using namespace solver_structures;

void print_usage(){
	cout << "Usage:"<<endl;
	cout << "-------------------------------------"<<endl;
	cout << "Required Parameters:"<<endl;

	cout << "-------------------------------------"<<endl;
	cout << "Optional Parameters:"<<endl;

}

int parse_console_options(solver_structures::optimization_settings* settings,
		int argc, char *argv[]) {

	char c;
	/*
	 * d - data file
	 * f - input file type
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
			settings->distributed_row_grid_file = atoi(optarg);
			break;
		case 's':
			settings->starting_points = atoi(optarg);
			break;
		case 'b':
			settings->batch_size = atoi(optarg);
			break;
		case 'n':
			settings->constrain = atoi(optarg);
			break;
		case 'm':
			settings->penalty= atof(optarg);
			break;
		case 'i':
			settings->max_it = atoi(optarg);
			break;
		case 'd':
			settings->data_file = optarg;
			data_file = true;
			break;
		case 'r':
			settings->result_file = optarg;
			result_file = true;
			break;
		case 'u':
			settings->on_the_fly_generation= atoi(optarg);
			break;
		case 't':
			settings->toll = atof(optarg);
			break;
		case 'p':
			settings->double_precission = true;
			break;
		case 'v':
			settings->verbose = true;
			break;
		case 'a':
			switch (atoi(optarg)) {
			case 1:
				settings->algorithm=L0_constrained_L1_PCA;
				algorithm=true;
				break;
			case 2:
				settings->algorithm=L0_constrained_L2_PCA;
				algorithm=true;
				break;
			case 3:
				settings->algorithm=L1_constrained_L1_PCA;
				algorithm=true;
				break;
			case 4:
				settings->algorithm=L1_constrained_L2_PCA;
				algorithm=true;
				break;
			case 5:
				settings->algorithm=L0_penalized_L1_PCA;
				algorithm=true;
				break;
			case 6:
				settings->algorithm=L0_penalized_L1_PCA;
				algorithm=true;
				break;
			case 7:
				settings->algorithm=L1_penalized_L1_PCA;
				algorithm=true;
				break;
			case 8:
				settings->algorithm=L1_penalized_L2_PCA;
				algorithm=true;
				break;

			}

			break;
		}
	}

	if (!data_file || !result_file || !algorithm) {
		if (settings->proccess_node==0)
			print_usage();
		return 1;
	}

	if (settings->batch_size > settings->starting_points) {
		settings->starting_points = settings->batch_size;
	}
	return 0;
}

#endif /* OPTION_PARSER_H_ */
