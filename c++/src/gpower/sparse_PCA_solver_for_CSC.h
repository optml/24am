/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M.Jahani, S. Damla Ahipasaoglu and M. Takac 
 *    "Alternating Maximization: Unified Framework and 24 Parallel Codes for L1 and L2 based Sparse PCA"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 * 
 */

#ifndef SPARSE_PCA_SOLVER_CSC_H_
#define SPARSE_PCA_SOLVER_CSC_H_
#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
#include "../utils/my_cblas_wrapper.h"
#include "my_sparse_cblas_wrapper.h"
#include "../utils/thresh_functions.h"
#include "../utils/timer.h"

#include "sparse_PCA_thresholding.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

template<typename F>
void printDescriptions(F* x, int length, const char* description,
		SolverStructures::OptimizationStatistics* optimizationStatistics,
		ofstream& stream) {
	FILE * fin = fopen(description, "r");
	char buffer[1000];
	if (fin == NULL) {
		printf("File not found\n");
		exit(1);
	} else {
		printf("------\n");
		cout << "Objective value: " << optimizationStatistics->fval << endl;
		stream << "Objective value: " << optimizationStatistics->fval << endl;
		for (int k = 0; k < length; k++) {
			fscanf(fin, "%s\n", &buffer);
			if (x[k] != 0) {
				stream << k << ":" << buffer << endl;
				printf("%d: %s\n", k, buffer);
			}

		}
		fclose(fin);
	}
}

namespace SPCASolver {

template<typename F>
class SparseDeflation {
public:
	std::vector<int> idx;
	std::vector<F> vals;
};

template<typename F>
class SparseDeflationCollection {
public:
	std::vector<SparseDeflation<F> > list;
	std::vector<F> buffer;

	void deflateVByPV(F* V, int n, int experiments, SparseDeflation<F> & pv) {
		// V = (I - xx') V
		F zero = 0.0;
		buffer.resize(experiments);
		cblas_vector_scale(experiments, &buffer[0], zero);
		for (int ex = 0; ex < experiments; ex++) {
			for (int cor = 0; cor < pv.idx.size(); cor++) {
				buffer[ex] += V[pv.idx[cor] + n * ex] * pv.vals[cor];
			}
		}

		for (int ex = 0; ex < experiments; ex++) {
			for (int cor = 0; cor < pv.idx.size(); cor++) {
				V[pv.idx[cor] + n * ex] -= buffer[ex] * pv.vals[cor];
			}
		}

	}

	void deflateV(F* V, int n, int experiments) {
		for (int pv = 0; pv < list.size(); pv++) {
			this->deflateVByPV(V, n, experiments, list[pv]);
		}
	}

	void addNewSparsePV(std::vector<F> &x) {
		SparseDeflation<F> sparseDeflation;
		sparseDeflation.idx.resize(0);
		sparseDeflation.vals.resize(0);
		for (int i = 0; i < x.size(); i++) {
			if (x[i] != 0) {
				sparseDeflation.vals.push_back(x[i]);
				sparseDeflation.idx.push_back(i);
			}
		}
		list.push_back(sparseDeflation);
	}
};

/*
 * Matrix B is stored in column order (Fortran Based)
 */
template<typename F>
F sparse_PCA_solver_CSC(F * B_CSC_Vals, int* B_CSC_Row_Id, int* B_CSC_Col_Ptr,
		F * x, int m, int n,
		SolverStructures::OptimizationSettings* optimizationSettings,
		SolverStructures::OptimizationStatistics* optimizationStatistics,
		bool doMean, F * means, bool doRowMean, F * rowMeans,
		SPCASolver::SparseDeflationCollection<F>& sparseDeflationCollection) {
	int number_of_experiments = optimizationSettings->totalStartingPoints;

	std::vector<ValueCoordinateHolder<F> > valsVec(number_of_experiments);
	ValueCoordinateHolder<F>* vals = &valsVec[0];

	std::vector<F> Zvec(m * number_of_experiments);
	F* Z = &Zvec[0];
	std::vector<F> Vvec(n * number_of_experiments);
	F* V = &Vvec[0];
	std::vector<F> ZZvec(m * number_of_experiments);
	F* ZZ = &ZZvec[0];
	std::vector<F> VVvec(n * number_of_experiments);
	F* VV = &VVvec[0];

	optimizationStatistics->it = optimizationSettings->maximumIterations;
	// Allocate vector for optimizationStatistics to return which point needs how much iterations
	if (optimizationSettings->storeIterationsForAllPoints) {
		optimizationStatistics->iters.resize(
				optimizationSettings->totalStartingPoints, -1);
	}
	F FLOATING_ZERO = 0;
	if (optimizationSettings->isConstrainedProblem()) {
		//				cblas_dscal(n * number_of_experiments, 0, V, 1);
#ifdef _OPENMP
//#pragma omp parallel for
#endif
		for (unsigned int j = 0; j < number_of_experiments; j++) {
			myseed = rand();
			F tmp_norm = 0;
			//			for (unsigned int i = 0; i < n;i++){//optimizationSettings->constraintParameter; i++) {
			//				unsigned int idx = i;

			for (unsigned int i = 0; i < n; i++) {
				unsigned int idx = i;//(int) (n * (F) rand_r(&myseed) / (RAND_MAX));
				if (idx == n)
					idx--;
				//printf("%d\n",idx);

				F tmp = (F) rand_r(&myseed) / RAND_MAX;
				V[j * n + idx] = tmp;
				tmp_norm += tmp * tmp;
			}
			cblas_vector_scale(n, &V[j * n], 1 / sqrt(tmp_norm));
		}

	} else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (unsigned int j = 0; j < number_of_experiments; j++) {
			myseed = j;
			F tmp_norm = 0;
			for (unsigned int i = 0; i < m; i++) {
				F tmp = (F) rand_r(&myseed) / RAND_MAX;
				tmp = -1 + 2 * tmp;
				Z[j * m + i] = tmp;
			}
		}
	}

	F error = 0;
	F max_errors[TOTAL_THREADS];

	F floating_zero = 0;
	F floating_one = 1;
	MKL_INT ONE_MKL_INT = 1;
	char matdescra[6];
	matdescra[0] = 'g';
	matdescra[1] = 'X';
	matdescra[2] = 'X';
	matdescra[3] = 'C';

	std::vector<std::vector<F> > bufferVector(number_of_experiments);
	std::vector<F>* buffer = &bufferVector[0];

	if (optimizationSettings->isConstrainedProblem()) {
		for (unsigned int j = 0; j < number_of_experiments; j++) {
			buffer[j].resize(n);
		}
	}

	double start_time_of_iterations = gettime();
	for (unsigned int it = 0; it < optimizationSettings->maximumIterations;
			it++) {
		for (unsigned int tmp = 0; tmp < TOTAL_THREADS; tmp++) {
			max_errors[tmp] = 0;
		}
		if (optimizationSettings->isConstrainedProblem()) {

			sparseDeflationCollection.deflateV(V, n, number_of_experiments);

			for (int ex = 0; ex < number_of_experiments; ex++) {
				for (int i = 0; i < n; i++)
					VV[i * number_of_experiments + ex] = V[i + ex * n];
			}
			sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_NOTRANS, m,
					number_of_experiments, n, &floating_one, matdescra,
					B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr, &B_CSC_Col_Ptr[1],
					VV, number_of_experiments, &floating_zero, ZZ,
					number_of_experiments);
			for (int ex = 0; ex < number_of_experiments; ex++) {
				for (int i = 0; i < m; i++)
					Z[i + m * ex] = ZZ[number_of_experiments * i + ex];
			}

			if (doMean) {
//				we have done   Z = B*V, now and we would like to have
//				Z = (B - E*diag(means) ) * V
				for (int ex = 0; ex < number_of_experiments; ex++) {
					F tmpVal = 0;
					for (int kk = 0; kk < n; kk++) {
						tmpVal += means[kk] * V[kk + n * ex];
					}
					for (int i = 0; i < m; i++) {
						Z[i + m * ex] -= tmpVal;
					}
				}
			}
			if (doRowMean) {
				//				we have done   Z = B*V, now and we would like to have
				//				Z = (B - diag(rowMeans) *E ) * V
				for (int ex = 0; ex < number_of_experiments; ex++) {
					F tmpVal = 0;
					for (int kk = 0; kk < n; kk++) {
						tmpVal += V[kk + n * ex];
					}
					for (int i = 0; i < m; i++) {
						Z[i + m * ex] -= tmpVal * rowMeans[i];
					}
				}
			}

			//set Z=sgn(Z)
			if (optimizationSettings->formulation
					== SolverStructures::L0_constrained_L1_PCA
					|| optimizationSettings->formulation
							== SolverStructures::L1_constrained_L1_PCA) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (unsigned int j = 0; j < number_of_experiments; j++) {
					vals[j].tmp = cblas_l1_norm(m, &Z[m * j], 1);
					vector_sgn(&Z[m * j], m);			//y=sgn(y)
				}
			}

			for (int ex = 0; ex < number_of_experiments; ex++) {
				for (int i = 0; i < m; i++)
					ZZ[number_of_experiments * i + ex] = Z[i + m * ex];
			}
			sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_TRANS, m,
					number_of_experiments, n, &floating_one, matdescra,
					B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr, &B_CSC_Col_Ptr[1],
					ZZ, number_of_experiments, &floating_zero, VV,
					number_of_experiments);
			for (int ex = 0; ex < number_of_experiments; ex++) {
				for (int i = 0; i < n; i++)
					V[i + ex * n] = VV[i * number_of_experiments + ex];
			}
			if (doMean) {
//				we have done   Z = B*V, now and we would like to have
//				V = (B - E*diag(means) )' * Z
//				V = B'Z - diag(means) E' * Z
				for (int ex = 0; ex < number_of_experiments; ex++) {
					F tmpVal = 0;
					for (int i = 0; i < m; i++) {
						tmpVal += Z[i + m * ex];
					}
					for (int kk = 0; kk < n; kk++) {
						V[kk + n * ex] -= means[kk] * tmpVal;
					}

//			     for (int kk = 0; kk < n; kk++) {
//					for (int i = 0; i < m; i++) {
//							V[kk + n * ex] -= means[kk] * Z[i + m * ex];
//						}
//					}
				}
			}
			if (doRowMean) {
				//				we have done   Z = B*V, now and we would like to have
				//				V = (B - diag(rowMeans) E* )' * Z
				for (int ex = 0; ex < number_of_experiments; ex++) {
					F tmpVal = 0;
					for (int i = 0; i < m; i++) {
						tmpVal += Z[i + m * ex] * rowMeans[i];
					}
					for (int kk = 0; kk < n; kk++) {
						V[kk + n * ex] -= tmpVal;
					}
				}
			}

			sparseDeflationCollection.deflateV(V, n, number_of_experiments);

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (unsigned int j = 0; j < number_of_experiments; j++) {
				F fval_current = 0;
				if (optimizationSettings->formulation
						== SolverStructures::L0_constrained_L2_PCA
						|| optimizationSettings->formulation
								== SolverStructures::L1_constrained_L2_PCA) {
					fval_current = cblas_l2_norm(m, &Z[m * j], 1);
				}
				F norm_of_x;
				if (optimizationSettings->isL1ConstrainedProblem()) {
					norm_of_x = soft_thresholding(&V[n * j], n,
							optimizationSettings->constraintParameter,
							buffer[j], optimizationSettings); // x = S_w(x)
				} else {
					norm_of_x = k_hard_thresholding(&V[n * j], n,
							optimizationSettings->constraintParameter,
							buffer[j], optimizationSettings); // x = T_k(x)
				}

				cblas_vector_scale(n, &V[j * n], 1 / norm_of_x);
				if (optimizationSettings->formulation
						== SolverStructures::L0_constrained_L1_PCA
						|| optimizationSettings->formulation
								== SolverStructures::L1_constrained_L1_PCA) {
					fval_current = vals[j].tmp;
				}
				F tmp_error = computeTheError(fval_current, vals[j].val,
						optimizationSettings);
				//Log end of iteration for given point
				if (optimizationSettings->storeIterationsForAllPoints
						&& termination_criteria(tmp_error, it,
								optimizationSettings)
						&& optimizationStatistics->iters[j] == -1) {
					optimizationStatistics->iters[j] = it;
				} else if (optimizationSettings->storeIterationsForAllPoints
						&& !termination_criteria(tmp_error, it,
								optimizationSettings)
						&& optimizationStatistics->iters[j] != -1) {
					optimizationStatistics->iters[j] = -1;
				}
				//---------------
				if (max_errors[my_thread_id] < tmp_error)
					max_errors[my_thread_id] = tmp_error;
				vals[j].val = fval_current;
			}
		} else {
			//scale Z

			if (optimizationSettings->formulation
					== SolverStructures::L0_penalized_L1_PCA
					|| optimizationSettings->formulation
							== SolverStructures::L1_penalized_L1_PCA) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (unsigned int j = 0; j < number_of_experiments; j++) {
					vector_sgn(&Z[m * j], m);
				}
			} else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (unsigned int j = 0; j < number_of_experiments; j++) {
					F tmp_norm = cblas_l2_norm(m, &Z[m * j], 1);
					cblas_vector_scale(m, &Z[j * m], 1 / tmp_norm);
				}
			}

			//----------------------------------------------
			for (int ex = 0; ex < number_of_experiments; ex++) {
				for (int i = 0; i < m; i++)
					ZZ[number_of_experiments * i + ex] = Z[i + m * ex];
			}
			sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_TRANS, m,
					number_of_experiments, n, &floating_one, matdescra,
					B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr, &B_CSC_Col_Ptr[1],
					ZZ, number_of_experiments, &floating_zero, VV,
					number_of_experiments);
			for (int ex = 0; ex < number_of_experiments; ex++) {
				for (int i = 0; i < n; i++)
					V[i + ex * n] = VV[i * number_of_experiments + ex];
			}

			if (doMean) {
				//				we have done   Z = B*V, now and we would like to have
				//				V = (B - E*diag(means) )' * Z
				//				V = B'Z - diag(means) E' * Z
				for (int ex = 0; ex < number_of_experiments; ex++) {
					for (int kk = 0; kk < n; kk++) {
						for (int i = 0; i < m; i++) {
							V[kk + n * ex] -= means[kk] * Z[i + m * ex];
						}
					}
				}
			}
			if (doRowMean) {
				//				we have done   Z = B*V, now and we would like to have
				//				V = (B - diag(rowMeans) E* )' * Z
				for (int ex = 0; ex < number_of_experiments; ex++) {
					F tmpVal = 0;
					for (int i = 0; i < m; i++) {
						tmpVal += Z[i + m * ex] * rowMeans[i];
					}
					for (int kk = 0; kk < n; kk++) {
						V[kk + n * ex] -= tmpVal;
					}
				}
			}

			sparseDeflationCollection.deflateV(V, n, number_of_experiments);
			//----------------------------------------------

			if (optimizationSettings->isL1PenalizedProblem()) {
				L1_penalized_thresholding(number_of_experiments, n, V,
						optimizationSettings, max_errors, vals,
						optimizationStatistics, it);
			} else {
				L0_penalized_thresholding(number_of_experiments, n, V,
						optimizationSettings, max_errors, vals,
						optimizationStatistics, it);
			}
//----------------------------------------
			sparseDeflationCollection.deflateV(V, n, number_of_experiments);
			for (int ex = 0; ex < number_of_experiments; ex++) {
				for (int i = 0; i < n; i++)
					VV[i * number_of_experiments + ex] = V[i + ex * n];
			}
			sparse_matrix_matrix_multiply(MY_SPARSE_WRAPPER_NOTRANS, m,
					number_of_experiments, n, &floating_one, matdescra,
					B_CSC_Vals, B_CSC_Row_Id, B_CSC_Col_Ptr, &B_CSC_Col_Ptr[1],
					VV, number_of_experiments, &floating_zero, ZZ,
					number_of_experiments);
			for (int ex = 0; ex < number_of_experiments; ex++) {
				for (int i = 0; i < m; i++)
					Z[i + m * ex] = ZZ[number_of_experiments * i + ex];
			}
			if (doMean) {
				//				we have done   Z = B*V, now and we would like to have
				//				Z = (B - E*diag(means) ) * V
				for (int ex = 0; ex < number_of_experiments; ex++) {
					for (int i = 0; i < m; i++) {
						for (int kk = 0; kk < n; kk++) {
							Z[i + m * ex] -= means[kk] * V[kk + n * ex];
						}
					}
				}
			}
			if (doRowMean) {
				//				we have done   Z = B*V, now and we would like to have
				//				Z = (B - diag(rowMeans) *E ) * V
				for (int ex = 0; ex < number_of_experiments; ex++) {
					F tmpVal = 0;
					for (int kk = 0; kk < n; kk++) {
						tmpVal += V[kk + n * ex];
					}
					for (int i = 0; i < m; i++) {
						Z[i + m * ex] -= tmpVal * rowMeans[i];
					}
				}
			}
			//-------------------------------------
		}
		error =
				max_errors[cblas_vector_max_index(TOTAL_THREADS, max_errors, 1)];
		if (termination_criteria(error, it, optimizationSettings)) {
			optimizationStatistics->it = it;
			break;
		}

	}
	double end_time_of_iterations = gettime();
//compute corresponding x
	optimizationStatistics->values.resize(
			optimizationSettings->totalStartingPoints);
	int selected_idx = 0;
	F best_value = vals[selected_idx].val;
	optimizationStatistics->values[0] = best_value;
	optimizationStatistics->totalTrueComputationTime = (end_time_of_iterations
			- start_time_of_iterations);
	for (unsigned int i = 1; i < number_of_experiments; i++) {
		optimizationStatistics->values[i] = vals[i].val;
		if (vals[i].val > best_value) {
			best_value = vals[i].val;
			selected_idx = i;
		}
	}
	cblas_vector_copy(n, &V[n * selected_idx], 1, x, 1);
	F norm_of_x = cblas_l2_norm(n, x, 1);
	cblas_vector_scale(n, x, 1 / norm_of_x); //Final x

	optimizationStatistics->fval = best_value;
	return best_value;
}

}

#endif /* SPARSE_PCA_SOLVER_H__ */
