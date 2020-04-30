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



#ifndef RANDOM_GENERATOR_H_
#define RANDOM_GENERATOR_H_


void generate_random_instance(int n, int m, gsl_matrix * B, gsl_matrix * BT,
		gsl_vector * x) {

#ifdef _OPENMP
	//#pragma omp parallel for
#endif
	for (int i = 0; i < n; i++) {
		gsl_vector_set(x, i, (double) rand_r(&myseed) / RAND_MAX);
		for (int j = 0; j < m; j++) {
			double tmp = (double) rand_r(&myseed) / RAND_MAX;
			tmp = tmp * 2 - 1;
			gsl_matrix_set(B, j, i, tmp);
		}
		gsl_vector_view col = gsl_matrix_column(B, i);
		double col_norm = gsl_blas_dnrm2(&col.vector);
		gsl_vector_scale(&col.vector, 1 / col_norm);
	}
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			gsl_matrix_set(BT, i, j, gsl_matrix_get(B, j, i));
		}
	}
}




#endif /* RANDOM_GENERATOR_H_ */
