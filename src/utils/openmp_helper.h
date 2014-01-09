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
 * openmp_helper.h
 *
 *  Created on: Mar 29, 2012
 *      Author: taki
 */

#ifndef OPENMP_HELPER_H
#define OPENMP_HELPER_H

#include <omp.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>





  unsigned int myseed = 0;
 unsigned int my_thread_id = 0;
 unsigned int TOTAL_THREADS=1;

#pragma omp threadprivate (my_thread_id)



void init_random_seeds() {
	TOTAL_THREADS=1;
#pragma omp parallel
	{
		TOTAL_THREADS = omp_get_num_threads();
	}
#ifdef DEBUG
	printf("Using %d threads\n",TOTAL_THREADS);
#endif
	unsigned int seed[TOTAL_THREADS];
#pragma omp parallel
	{
		my_thread_id =omp_get_thread_num();
		if (omp_get_thread_num()==0) {
			srand(1);
			for (unsigned int i = 0; i < TOTAL_THREADS; i++)
			seed[i] = (unsigned int) RAND_MAX * rand();
		}
	}
#pragma omp parallel
	{
		myseed = seed[omp_get_thread_num()];
	}
}


#endif /* OPENMP_HELPER_H_ */
