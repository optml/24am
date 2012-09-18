
#ifndef GSL_VARIOUS_H_
#define GSL_VARIOUS_H_

#include "gsl_helper.h"





void getFileSize(const char* filename, int& DIM_M, int& DIM_N) {
	FILE * fin = fopen(filename, "r");
	if (fin == NULL) {

	} else {
		int status = fscanf(fin, "%d;%d", &DIM_M, &DIM_N);
		if (status) exit(0);
		fclose(fin);
	}
}

void readFromFile(const char* filename, int& DIM_M, int& DIM_N, gsl_matrix * B,
		gsl_matrix * BT) {
	int i, j;
	FILE * fin = fopen(filename, "r");
	if (fin == NULL) {
	} else {
		int status = fscanf(fin, "%d;%d", &DIM_M, &DIM_N);
		for (j = 0; j < DIM_M; j++) {
			for (i = 0; i < DIM_N; i++) {
				float tmp = -1;
				status = fscanf(fin, "%f;", &tmp);
				gsl_matrix_set(B, j, i, tmp);
				gsl_matrix_set(BT, i, j, tmp);
			}
		}
		fclose(fin);
	}
}


double computeReferentialValue(gsl_matrix * B, gsl_vector * x, gsl_vector * y) {
	gsl_blas_dgemv(CblasNoTrans, 1, B, x, 0.0, y); // Multiply y = B*x
	double fval2 = gsl_blas_dnrm2(y);
	return fval2 * fval2;
}


#endif /* VARIOUS_H_ */
