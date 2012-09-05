/*
 * gpower_problem_generator.h
 *
 *  Created on: Apr 29, 2012
 *      Author: taki
 */

#ifndef GPOWER_PROBLEM_GENERATOR_H_
#define GPOWER_PROBLEM_GENERATOR_H_





template<typename F>
int generateProblem(const int n, const int m,thrust::host_vector<F>& h_B) {

	printf("%d x %d \n", m, n);

	char final_file[1000];
	unsigned int seed = 0;
	for (int i = 0; i < n; i++) {
		F total = 0;
		for (int j = 0; j < m; j++) {
			F tmp = (F) rand_r(&seed) / RAND_MAX;
			tmp=-1+2*tmp;
			h_B[j+i*m]=tmp;
			total+=tmp*tmp;
		}
		total=sqrt(total);
		for (int j = 0; j < m; j++) {
			h_B[j+i*m]=h_B[j+i*m]/total;
		}
	}
	return 0;
}

//template<typename F>
//int generateProblem(const int n, const int m,thrust::host_vector<F>& h_B,const unsigned int LD_M,const unsigned int LD_N) {
//
//	printf("%d x %d \n", m, n);
//
//	//	char final_file[1000];
//	unsigned int seed = 0;
//	for (int i = 0; i < n; i++) {
//		F total = 0;
//		for (int j = 0; j < m; j++) {
//			F tmp = (F) rand_r(&seed) / RAND_MAX;
//			tmp=-1+2*tmp;
//			h_B[j+i*LD_M]=tmp;
//			total+=tmp*tmp;
//		}
//		total=sqrt(total);
//		for (int j = 0; j < m; j++) {
//			h_B[j+i*LD_M]=h_B[j+i*LD_M]/total;
//		}
//	}
//	return 0;
//}


unsigned int vector_get_nnz(const double * x, int size) {
	unsigned int nnz = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:nnz)
#endif
	for (unsigned int i = 0; i < size; i++) {
		if (x[i] != 0)
			nnz++;
	}
	return nnz;
}















#endif /* GPOWER_PROBLEM_GENERATOR_H_ */
