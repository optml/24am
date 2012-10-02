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
 */


#ifndef GPU_HEADERS_H_
#define GPU_HEADERS_H_

#include <iostream>
#include <stdio.h>
#include "../utils/timer.h"
#include "../utils/various.h"

#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"

#include "thrust_headers.h"
#include "my_cublas_wrapper.h"
#include "gpu_sparse_PCA_solver.h"
#include "gpower_problem_generator.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

//template<typename F>
//void readFromFile(const char* filename, int& DIM_M, int& DIM_N,
//		thrust::host_vector<F> &B) {
//	int i, j;
//	FILE * fin = fopen(filename, "r");
//	if (fin == NULL) {
//	} else {
//		fscanf(fin, "%d;%d", &DIM_M, &DIM_N);
//		for (j = 0; j < DIM_M; j++) {
//			for (i = 0; i < DIM_N; i++) {
//				float tmp = -1;
//				fscanf(fin, "%f;", &tmp);
//				float asdf = (float) rand() / RAND_MAX;
//				if (asdf > 0.5)
//					asdf = -asdf;
//				B[IDX2C(j,i,DIM_M)] = tmp;
//			}
//		}
//		fclose(fin);
//	}
//}

#endif /* GPU_HEADERS_H_ */
