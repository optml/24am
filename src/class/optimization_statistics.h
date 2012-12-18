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
 *
 *  Created on: Mar 29, 2012
 *      Author: taki
 *
 *
 *   This class is used to store statistics during solver
 *
 */

#ifndef OPTIMIZATION_STATISTICS_H_
#define OPTIMIZATION_STATISTICS_H_

#include <vector>
namespace solver_structures {
class optimization_statistics {

public:
	unsigned int it; // total iterations solver did
	double fval; // final objective value
	double error; // final error
	double true_computation_time; // elapsed time used in AM method
	double total_elapsed_time; // total elapsed time including data loading and storing result
	unsigned int n; // size of problem (length of vector "x")
	std::vector<double> values; // final values for different starting point. This is used only for
	                             // creating some figures in paper.
	std::vector<int> iters; // the same as before, but holds the number of iterations for given starting point
	std::vector<int> cardinalities; // cardinalities for given starting point. For L0 constrained method doesn't make sense as
	                                // it's value has to be constrain parameter from settings
	int total_threads_used;
	optimization_statistics() {
		it = 0;
		total_threads_used=1;
	}
};
}
#endif /* OPTIMIZATION_STATISTICS_H_ */
