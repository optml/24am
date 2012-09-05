
#ifndef VARIOUS_H_
#define VARIOUS_H_

#include "gsl_helper.h"

template<typename F>
class value_coordinate_holder {
public:
	F val;
	F prev_val;
	F tmp;
	F current_error;
	bool done;
	unsigned int idx;
	value_coordinate_holder() {
		val = 0;
		tmp=0;
		done=false;
	}

	void reset(){
		val=0;
		prev_val=0;
		tmp=0;
		done=true;
	}

};

template<typename T> int sgn(T val) {
	return (val > T(0)) - (val < T(0));
}


template<typename F>
unsigned int vector_get_nnz(F * x, int length) {
	unsigned int nnz = 0;
	for (unsigned int i = 0; i < length; i++) {
		if (x[i] != 0)
			nnz++;
	}
	return nnz;
}


void getFileSize(const char* filename, int& DIM_M, int& DIM_N) {
	FILE * fin = fopen(filename, "r");
	if (fin == NULL) {

	} else {
		fscanf(fin, "%d;%d", &DIM_M, &DIM_N);
		fclose(fin);
	}
}

void readFromFile(const char* filename, int& DIM_M, int& DIM_N, gsl_matrix * B,
		gsl_matrix * BT) {
	int i, j;
	FILE * fin = fopen(filename, "r");
	if (fin == NULL) {
	} else {
		fscanf(fin, "%d;%d", &DIM_M, &DIM_N);
		for (j = 0; j < DIM_M; j++) {
			for (i = 0; i < DIM_N; i++) {
				float tmp = -1;
				fscanf(fin, "%f;", &tmp);
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
