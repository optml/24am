/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M.Jahani, S. Damla Ahipasaoglu and M. Takac
 *    "Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 * 
 */


#ifndef DATA_GENERATION_H_
#define DATA_GENERATION_H_

#include "../utils/matrix_conversions.h"

template<typename F>
int load_doc_data(const char* filename, int& n, int& m, std::vector<F>& B) {
	FILE * fin = fopen(filename, "r");
	int nnz = -1;
	int row = -1;
	int col = -1;
	int val = -1;
	if (fin == NULL) {
		printf("File not found\n");
		exit(1);
	} else {
		fscanf(fin, "%d", &m);
		fscanf(fin, "%d", &n);
		fscanf(fin, "%d", &nnz);
		B.resize(m * n, 0);
		for (int k = 0; k < nnz; k++) {
			fscanf(fin, "%d %d %d\n", &row, &col, &val);
			row--;
			col--;
			B[row + col * m] = val;

		}
		//		for (j = 0; j < DIM_M; j++) {
		//			for (i = 0; i < DIM_N; i++) {
		//				float tmp = -1;
		//				fscanf(fin, "%f;", &tmp);
		//				gsl_matrix_set(B, j, i, tmp);
		//				gsl_matrix_set(BT, i, j, tmp);
		//			}
		//		}
		fclose(fin);
	}

	for (col = 0; col < n; col++) {
		//		F mean = 0;
		//		for (row = 0; row < m; row++) {
		//			mean += B[row + col * m];
		//		}
		//		mean = mean / (0.0 + m);
		//		for (row = 0; row < m; row++) {
		//			B[row + col * m] = B[row + col * m] - mean;
		//		}

		F norm = 0;
		for (row = 0; row < m; row++) {
			norm += B[row + col * m] * B[row + col * m];
		}
		if (norm > 0) {
			norm = sqrt(norm);
			for (row = 0; row < m; row++) {
				F tmp = B[row + col * m];
				B[row + col * m] = tmp / norm;
				//				if (tmp!=0)
				//					printf("%f ",tmp / norm);
			}
		}
	}

	//	exit(0);
	//	printf("%d x %d \n", m, n);
	//
	//	char final_file[1000];
	//	unsigned int seed = 0;
	//	for (int i = 0; i < n; i++) {
	//		F total = 0;
	//		for (int j = 0; j < m; j++) {
	//			F tmp = (F) rand_r(&seed) / RAND_MAX;
	//			tmp = -1 + 2 * tmp;
	//			h_B[j + i * m] = tmp;
	//			total += tmp * tmp;
	//		}
	//		total = sqrt(total);
	//		for (int j = 0; j < m; j++) {
	//			h_B[j + i * m] = h_B[j + i * m] / total;
	//		}
	//	}


	return 0;
}

template<typename F, typename I>
int load_doc_data(const char* filename, int& n, int& m,
		std::vector<F>& B_CSC_Vals, std::vector<I> &B_CSC_Row_Id,
		std::vector<I> &B_CSC_Col_Ptr, std::vector<F>& means, bool doMeans) {
	FILE * fin = fopen(filename, "r");
	int nnz = -1;
	int row = -1;
	int col = -1;
	int val = -1;
	if (fin == NULL) {
		printf("File not found\n");
		exit(1);
	} else {
		fscanf(fin, "%d", &m);
		fscanf(fin, "%d", &n);
		fscanf(fin, "%d", &nnz);

		std::vector<F> B_COO_Vals(nnz, 0);
		std::vector<I> B_COO_Col_Id(nnz, 0);
		std::vector<I> B_COO_Row_Id(nnz, 0);

		for (int k = 0; k < nnz; k++) {
			fscanf(fin, "%d %d %d\n", &row, &col, &val);
			row--;
			col--;

			B_COO_Row_Id[k] = row;
			B_COO_Col_Id[k] = col;
			B_COO_Vals[k] = val;
		}
		//		for (j = 0; j < DIM_M; j++) {
		//			for (i = 0; i < DIM_N; i++) {
		//				float tmp = -1;
		//				fscanf(fin, "%f;", &tmp);
		//				gsl_matrix_set(B, j, i, tmp);
		//				gsl_matrix_set(BT, i, j, tmp);
		//			}
		//		}
		fclose(fin);
		getCSC_from_COO(B_COO_Vals, B_COO_Row_Id, B_COO_Col_Id, B_CSC_Vals,
				B_CSC_Row_Id, B_CSC_Col_Ptr, m, n);

		//		for (int i=0;i<nnz;i++){
		//			printf("ZZZIII %d %d %f\n",B_CSC_Row_Id[i],B_CSC_Col_Ptr[i],B_CSC_Vals[i]);
		//		}
		//		printf("ZZZIII %d \n",B_CSC_Col_Ptr[n]);


	}

	if (doMeans) {
		means.resize(n);
		for (col = 0; col < n; col++) {
			means[col] = 0;
			for (row = B_CSC_Col_Ptr[col]; row < B_CSC_Col_Ptr[col + 1]; row++) {
				means[col] += B_CSC_Vals[row];
			}
			means[col] = means[col] / (0.0 + m);
			for (row = B_CSC_Col_Ptr[col]; row < B_CSC_Col_Ptr[col + 1]; row++) {
				B_CSC_Vals[row] = B_CSC_Vals[row] - means[col];
			}
		}
	}

	for (col = 0; col < n; col++) {
//		F norm = 0;
//		for (row = B_CSC_Col_Ptr[col]; row < B_CSC_Col_Ptr[col + 1]; row++) {
//			norm += B_CSC_Vals[row] * B_CSC_Vals[row];
//		}
//
//		if (doMeans) {
//			norm += means[col] * means[col] * (m - B_CSC_Col_Ptr[col + 1]
//					- B_CSC_Col_Ptr[col]);
//		}
//		if (norm > 0) {
//			norm = sqrt(norm);
//			for (row = B_CSC_Col_Ptr[col]; row < B_CSC_Col_Ptr[col + 1]; row++) {
//				B_CSC_Vals[row] = B_CSC_Vals[row] / norm;
//			}
//		}
		means[col]=-means[col];
	}

	return 0;
}

#endif /* DATA_GENERATION_H_ */
