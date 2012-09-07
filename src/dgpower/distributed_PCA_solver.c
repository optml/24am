/*
 * distributed_pca_solver.h
 *
 *  Created on: Mar 30, 2012
 *      Author: taki
 */

#include "distributed_PCA_solver.h"

int distributed_pca_solver(char* filename, char* outputfile, int MAP_X,
		int input_files, struct optimization_settings* settings,
		struct optimization_statistics* stat) {
	MKL_INT ROW_BLOCKING = 64;
	MKL_INT COL_BLOCKING = ROW_BLOCKING;
	MKL_INT X_VECTOR_BLOCKING = ROW_BLOCKING;
	MKL_INT iam, nprocs, ictxt, ictxt2, myrow, mycol, nprow, npcol;
	MKL_INT info;
	MKL_INT m, n, nb, mb, mp, nq, lld, lld_local;
	int i, j, k;
	blacs_pinfo_(&iam, &nprocs);
	blacs_get_(&i_negone, &i_zero, &ictxt);
	int MAP_Y = nprocs / MAP_X;
	if (MAP_X * MAP_Y != nprocs || input_files > MAP_X) {
		if (iam == 0)
			printf("Wrong Grid Map specification!  %d %d\n", MAP_X, MAP_Y);
		return -1;
	}
	blacs_gridinit_(&ictxt, "C", &MAP_X, &MAP_Y); // Create row map
	blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);
	/* ===========================================================================================
	 *                LOAD DATA FROM FILES AND DISTRIBUTE IT ACROS NODES
	 *
	 * ===========================================================================================
	 */
	// Load data from files
	char final_file[1000];
	sprintf(final_file, "%s%d", filename, iam);
	int DIM_M = 0;
	int DIM_N = 0;
	int DIM_N_LOCAL = 0;
	int DIM_M_LOCAL = 0;
	double * B_Local;

	double start_time = gettime();

	FILE * fin = fopen(final_file, "r");
	if (fin == NULL) {
		B_Local = (double*) calloc(0, sizeof(double));
	} else {
		fscanf(fin, "%d;%d", &DIM_M_LOCAL, &DIM_N_LOCAL);
		B_Local = (double*) calloc(DIM_M_LOCAL * DIM_N_LOCAL, sizeof(double));
		for (j = 0; j < DIM_M_LOCAL; j++) {
			for (i = 0; i < DIM_N_LOCAL; i++) {
				float tmp = -1;
				fscanf(fin, "%f;", &tmp);
				B_Local[i * DIM_M_LOCAL + j] = tmp;
			}
		}
		fclose(fin);
	}
	double end_time = gettime();
	if (iam == 0) {
		printf("loading data from file into memmory took %f\n",
				end_time - start_time);
	}

	//-------------------------------

	start_time = gettime();

	i_tmp1 = -1, i_tmp2 = -1;
	DIM_N = DIM_N_LOCAL;
	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_N, &i_one, &i_tmp1, &i_tmp2, &i_one, &i_negone,
			&i_negone);
	DIM_M = DIM_M_LOCAL;
	igsum2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_M, &i_one, &i_negone, &i_negone);
	MKL_INT B_Local_row_blocking = DIM_M_LOCAL;
	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &B_Local_row_blocking, &i_one, &i_tmp1, &i_tmp2, &i_one,
			&i_negone, &i_negone);

	// Now, the size of Matrix B is   DIM_M x DIM_n
	/*  Matrix descriptors */
	MDESC descB, descB_local;
	/* Create Local Descriptors + Global Descriptors*/
	i_tmp1 = numroc_(&DIM_M, &B_Local_row_blocking, &myrow, &i_zero, &nprow);
	i_tmp1 = MAX(1, i_tmp1);
	descinit_(descB_local, &DIM_M, &DIM_N, &B_Local_row_blocking, &DIM_N,
			&i_zero, &i_zero, &ictxt, &i_tmp1, &info);

	mp = numroc_(&DIM_M, &ROW_BLOCKING, &myrow, &i_zero, &nprow);
	nq = numroc_(&DIM_N, &COL_BLOCKING, &mycol, &i_zero, &npcol);
	lld = MAX(mp, 1);
	descinit_(descB, &DIM_M, &DIM_N, &ROW_BLOCKING, &COL_BLOCKING, &i_zero,
			&i_zero, &ictxt, &lld, &info);

	end_time = gettime();
	if (iam == 0) {
		printf("allocate descriptors and vectors %f\n", end_time - start_time);
	}

	// Distribute data from BLocal => B
	start_time = gettime();
	double * B = (double*) calloc(mp * nq, sizeof(double));
	pdgeadd_(&transNo, &DIM_M, &DIM_N, &one, B_Local, &i_one, &i_one,
			descB_local, &zero, B, &i_one, &i_one, descB);
	free(B_Local);

	end_time = gettime();
	if (iam == 0) {
		printf("matrix distribution accross the grid took %f\n",
				end_time - start_time);
	}

	/* =============================================================
	 *         Initialize vector "x" where solution will be stored
	 *
	 * =============================================================
	 */

	MKL_INT x_mp = numroc_(&DIM_N, &X_VECTOR_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT x_nq = numroc_(&i_one, &X_VECTOR_BLOCKING, &mycol, &i_zero, &npcol);
	double * x = (double*) calloc(x_mp * x_nq, sizeof(double));
	MDESC desc_x;
	int nnz_x = x_mp * x_nq;
	i_tmp1 = MAX(1, x_mp);
	descinit_(desc_x, &DIM_N, &i_one, &X_VECTOR_BLOCKING, &X_VECTOR_BLOCKING,
			&i_zero, &i_zero, &ictxt, &i_tmp1, &info);
	/* =============================================================
	 *          CALL SOLVER
	 * =============================================================
	 */

	start_time = gettime();
	distributed_sparse_PCA_solver(B, x, descB, nnz_x, ictxt, ROW_BLOCKING,
			X_VECTOR_BLOCKING, settings, stat);

	end_time = gettime();
	if (iam == 0) {
		printf("time spend in solver %f\n", end_time - start_time);
	}
	/* =============================================================
	 *          STORE RESULT
	 * =============================================================
	 */

	start_time = gettime();

	double* x_local = NULL;
	if (iam == 0) {
		x_local = (double*) calloc(DIM_N, sizeof(double));
	} else {
		x_local = (double*) calloc(0, sizeof(double));
	}
	MDESC desc_x_local;
	MKL_INT x_local_mq = numroc_(&DIM_N, &DIM_N, &myrow, &i_zero, &nprow);
	MKL_INT x_local_np = numroc_(&i_one, &i_one, &mycol, &i_zero, &npcol);
	i_tmp1 = MAX(1, x_local_mq);
	descinit_(desc_x_local, &DIM_N, &i_one, &DIM_N, &i_one, &i_zero, &i_zero,
			&ictxt, &i_tmp1, &info);
	pdgeadd_(&transNo, &DIM_N, &i_one, &one, x, &i_one, &i_one, desc_x, &zero,
			x_local, &i_one, &i_one, desc_x_local);
	if (iam == 0) {
		fin = fopen(outputfile, "w");
		for (i = 0; i < DIM_N; i++) {
			fprintf(fin, "%f;", x_local[i]);
		}
		fclose(fin);
	}

	end_time = gettime();
	if (iam == 0) {
		printf("stooring result %f\n", end_time - start_time);
	}

	free(x_local);
	free(x);
	free(B);
	blacs_gridexit_(&ictxt);
	blacs_exit_(&i_zero);
	return iam;
}

int distributed_pca_solver_from_two_dim_files(char* filename, char* outputfile,
		int MAP_X, struct optimization_settings* settings,
		struct optimization_statistics* stat) {
	MKL_INT ROW_BLOCKING = 64;
	MKL_INT COL_BLOCKING = ROW_BLOCKING;

	MKL_INT X_VECTOR_BLOCKING = ROW_BLOCKING;
	MKL_INT iam, nprocs, ictxt, ictxt2, myrow, mycol, nprow, npcol;
	MKL_INT info;
	MKL_INT m, n, nb, mb, mp, nq, lld, lld_local;
	int i, j, k;
	blacs_pinfo_(&iam, &nprocs);
	blacs_get_(&i_negone, &i_zero, &ictxt);
	int MAP_Y = nprocs / MAP_X;
	if (MAP_X * MAP_Y != nprocs) {
		if (iam == 0)
			printf("Wrong Grid Map specification!  %d %d\n", MAP_X, MAP_Y);
		return -1;
	}
	blacs_gridinit_(&ictxt, "C", &MAP_X, &MAP_Y); // Create row map
	blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);
	/* ===========================================================================================
	 *                LOAD DATA FROM FILES AND DISTRIBUTE IT ACROS NODES
	 *
	 * ===========================================================================================
	 */
	// Load data from files
	char final_file[1000];
	sprintf(final_file, "%s%d-%d", filename, myrow, mycol);
	int DIM_M = 0;
	int DIM_N = 0;
	int DIM_N_LOCAL = 0;
	int DIM_M_LOCAL = 0;
	double * B_Local;
	blacs_barrier_(&ictxt, &C_CHAR_SCOPE_ALL);
	double start_time = gettime();

	FILE * fin = fopen(final_file, "r");
	if (fin == NULL) {
		B_Local = (double*) calloc(0, sizeof(double));
	} else {
		fscanf(fin, "%d;%d", &DIM_M_LOCAL, &DIM_N_LOCAL);
		B_Local = (double*) calloc(DIM_M_LOCAL * DIM_N_LOCAL, sizeof(double));
		for (j = 0; j < DIM_M_LOCAL; j++) {
			for (i = 0; i < DIM_N_LOCAL; i++) {
				float tmp = -1;
				fscanf(fin, "%f;", &tmp);
				B_Local[i * DIM_M_LOCAL + j] = tmp;
			}
		}
		fclose(fin);
	}
	double end_time = gettime();
	if (iam == 0) {
		printf("loading data from file into memmory took %f\n",
				end_time - start_time);
	}

	//-------------------------------
	blacs_barrier_(&ictxt, &C_CHAR_SCOPE_ALL);
	start_time = gettime();

	i_tmp1 = -1, i_tmp2 = -1;
	DIM_N = DIM_N_LOCAL;
	int DIM_N_INPUT_BLOCKING = DIM_N_LOCAL;
	igsum2d_(&ictxt, &C_CHAR_SCOPE_ROWS, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_N, &i_one, &i_tmp1, &i_tmp2, &i_one, &i_negone,
			&i_negone);

	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_N, &i_one, &i_tmp1, &i_tmp2, &i_one, &i_negone,
			&i_negone);

	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_N_INPUT_BLOCKING, &i_one, &i_tmp1, &i_tmp2, &i_one,
			&i_negone, &i_negone);

	DIM_M = DIM_M_LOCAL;
	igsum2d_(&ictxt, &C_CHAR_SCOPE_COLS, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_M, &i_one, &i_negone, &i_negone);
	MKL_INT DIM_MM = DIM_M;
	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &DIM_MM, &i_one, &i_tmp1, &i_tmp2, &i_one, &i_negone,
			&i_negone);
	DIM_M = DIM_MM;

	MKL_INT B_Local_row_blocking = DIM_M_LOCAL;
	igamx2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER, &i_one,
			&i_one, &B_Local_row_blocking, &i_one, &i_tmp1, &i_tmp2, &i_one,
			&i_negone, &i_negone);

	// Now, the size of Matrix B is   DIM_M x DIM_n
	/*  Matrix descriptors */
	MDESC descB, descB_local;
	/* Create Local Descriptors + Global Descriptors*/
	i_tmp1 = numroc_(&DIM_M, &B_Local_row_blocking, &myrow, &i_zero, &nprow);
	i_tmp1 = MAX(1, i_tmp1);
	descinit_(descB_local, &DIM_M, &DIM_N, &B_Local_row_blocking,
			&DIM_N_INPUT_BLOCKING, &i_zero, &i_zero, &ictxt, &i_tmp1, &info);

	mp = numroc_(&DIM_M, &ROW_BLOCKING, &myrow, &i_zero, &nprow);
	nq = numroc_(&DIM_N, &COL_BLOCKING, &mycol, &i_zero, &npcol);
	lld = MAX(mp, 1);
	descinit_(descB, &DIM_M, &DIM_N, &ROW_BLOCKING, &COL_BLOCKING, &i_zero,
			&i_zero, &ictxt, &lld, &info);

	end_time = gettime();
	if (iam == 0) {
		printf("allocate descriptors and vectors %f\n", end_time - start_time);
	}

	// Distribute data from BLocal => B
	blacs_barrier_(&ictxt, &C_CHAR_SCOPE_ALL);
	start_time = gettime();
	double * B = (double*) calloc(mp * nq, sizeof(double));
	pdgeadd_(&transNo, &DIM_M, &DIM_N, &one, B_Local, &i_one, &i_one,
			descB_local, &zero, B, &i_one, &i_one, descB);
	free(B_Local);

	end_time = gettime();
	if (iam == 0) {
		printf("matrix distribution accross the grid took %f\n",
				end_time - start_time);
	}

	/* =============================================================
	 *         Initialize vector "x" where solution will be stored
	 *
	 * =============================================================
	 */

	MKL_INT x_mp = numroc_(&DIM_N, &X_VECTOR_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT x_nq = numroc_(&i_one, &X_VECTOR_BLOCKING, &mycol, &i_zero, &npcol);
	double * x = (double*) calloc(x_mp * x_nq, sizeof(double));
	MDESC desc_x;
	int nnz_x = x_mp * x_nq;
	i_tmp1 = MAX(1, x_mp);
	descinit_(desc_x, &DIM_N, &i_one, &X_VECTOR_BLOCKING, &X_VECTOR_BLOCKING,
			&i_zero, &i_zero, &ictxt, &i_tmp1, &info);
	/* =============================================================
	 *          CALL SOLVER
	 * =============================================================
	 */

	settings->starting_points = 1;
			start_time = gettime();
			distributed_sparse_PCA_solver(B, x, descB, nnz_x, ictxt, ROW_BLOCKING,
					X_VECTOR_BLOCKING, settings, stat);

			end_time = gettime();

int it;
	for (it = 1; it <= 32; it = it * 4) {
		settings->max_it = it;


		blacs_barrier_(&ictxt, &C_CHAR_SCOPE_ALL);


		settings->starting_points = 1;
		start_time = gettime();
		distributed_sparse_PCA_solver(B, x, descB, nnz_x, ictxt, ROW_BLOCKING,
				X_VECTOR_BLOCKING, settings, stat);

		end_time = gettime();
		if (iam == 0) {
			printf("time spend in solver %f; stating points%d; it%d\n",
					end_time - start_time, settings->starting_points,
					settings->max_it);
		}


		blacs_barrier_(&ictxt, &C_CHAR_SCOPE_ALL);
		settings->starting_points = 32;
		start_time = gettime();
		distributed_sparse_PCA_solver(B, x, descB, nnz_x, ictxt, ROW_BLOCKING,
				X_VECTOR_BLOCKING, settings, stat);

		end_time = gettime();
		if (iam == 0) {
			printf("time spend in solver %f; stating points%d; it%d\n",
					end_time - start_time, settings->starting_points,
					settings->max_it);

		}


		blacs_barrier_(&ictxt, &C_CHAR_SCOPE_ALL);
		settings->starting_points = 64;
		start_time = gettime();
		distributed_sparse_PCA_solver(B, x, descB, nnz_x, ictxt, ROW_BLOCKING,
				X_VECTOR_BLOCKING, settings, stat);

		end_time = gettime();
		if (iam == 0) {
			printf("time spend in solver %f; stating points%d; it%d\n",
					end_time - start_time, settings->starting_points,
					settings->max_it);

		}
	}

	/* =============================================================
	 *          STORE RESULT
	 * =============================================================
	 */

	start_time = gettime();

	double* x_local = NULL;
	if (iam == 0) {
		x_local = (double*) calloc(DIM_N, sizeof(double));
	} else {
		x_local = (double*) calloc(0, sizeof(double));
	}
	MDESC desc_x_local;
	MKL_INT x_local_mq = numroc_(&DIM_N, &DIM_N, &myrow, &i_zero, &nprow);
	MKL_INT x_local_np = numroc_(&i_one, &i_one, &mycol, &i_zero, &npcol);
	i_tmp1 = MAX(1, x_local_mq);
	descinit_(desc_x_local, &DIM_N, &i_one, &DIM_N, &i_one, &i_zero, &i_zero,
			&ictxt, &i_tmp1, &info);
	pdgeadd_(&transNo, &DIM_N, &i_one, &one, x, &i_one, &i_one, desc_x, &zero,
			x_local, &i_one, &i_one, desc_x_local);
	if (iam == 0) {
		fin = fopen(outputfile, "w");
		for (i = 0; i < DIM_N; i++) {
			fprintf(fin, "%f;", x_local[i]);
		}
		fclose(fin);
	}

	end_time = gettime();
	if (iam == 0) {
		printf("stooring result %f\n", end_time - start_time);
	}

	free(x_local);
	free(x);
	free(B);
	blacs_gridexit_(&ictxt);
	blacs_exit_(&i_zero);
	return iam;
}


int sgn(double val) {
	return (val > (0)) - (val < (0));
}

double myabs(double val) {
	return (val > (0)) ? val : -val;
}

void clear_local_vector(double * v, const int n) {
	int i;
	for (i = 0; i < n; i++)
		v[i] = 0;
}

int get_column_coordinate(const int col, const int myCol, const int numCol,
		const int blocking) {
	const int fillup = col / blocking;
	return (fillup * (numCol - 1) + myCol) * blocking + col;
}

void distributed_sparse_PCA_solver(double * B, double *x, MDESC descB,
		int nnz_x, MKL_INT ictxt, MKL_INT ROW_BLOCKING,
		MKL_INT X_VECTOR_BLOCKING, struct optimization_settings* settings,
		struct optimization_statistics* stat) {
	MKL_INT myrow, mycol, nprow, npcol, info;
	int i, j;
	// create vector "z"
	blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);
	MKL_INT M = descB[2];
	MKL_INT N = descB[3];
	MKL_INT SP_MKL = settings->starting_points;
	MKL_INT z_mp = numroc_(&M, &ROW_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT z_nq = numroc_(&SP_MKL, &ROW_BLOCKING, &mycol, &i_zero, &npcol);

	unsigned int seed = mycol * nprow + myrow;
	const int nnz_z = z_mp * z_nq;
	double * z = (double*) calloc(nnz_z, sizeof(double));
	for (i = 0; i < nnz_z; i++) {
		z[i] = -1 + 2 * (double) rand_r(&seed) / RAND_MAX;
	}
	MDESC desc_z;
	MKL_INT i_tmp1 = MAX(1, z_mp);
	descinit_(desc_z, &M, &SP_MKL, &ROW_BLOCKING, &ROW_BLOCKING, &i_zero,
			&i_zero, &ictxt, &i_tmp1, &info);

	//=============== Create description for "V"
	MKL_INT V_mp = numroc_(&N, &X_VECTOR_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT V_nq =
			numroc_(&SP_MKL, &X_VECTOR_BLOCKING, &mycol, &i_zero, &npcol);
	MDESC desc_V;
	i_tmp1 = MAX(1, V_mp);
	descinit_(desc_V, &N, &SP_MKL, &X_VECTOR_BLOCKING, &X_VECTOR_BLOCKING,
			&i_zero, &i_zero, &ictxt, &i_tmp1, &info);
	double * V = (double*) calloc(V_mp * V_nq, sizeof(double));
	//=============== Create description for "x"
	MKL_INT x_mp = numroc_(&N, &X_VECTOR_BLOCKING, &myrow, &i_zero, &nprow);
	MKL_INT x_nq = numroc_(&i_one, &X_VECTOR_BLOCKING, &mycol, &i_zero, &npcol);
	MDESC desc_x;
	i_tmp1 = MAX(1, x_mp);
	descinit_(desc_x, &N, &i_one, &X_VECTOR_BLOCKING, &X_VECTOR_BLOCKING,
			&i_zero, &i_zero, &ictxt, &i_tmp1, &info);
	//=============== Create vector for "norms"
	double * norms =
			(double*) calloc(settings->starting_points, sizeof(double));
	struct value_coordinate_holder* values =
			(struct value_coordinate_holder*) calloc(settings->starting_points,
					sizeof(struct value_coordinate_holder));
	// ======================== RUN SOLVER
	stat->it = 0;
	double fval = 0;
	double fval_prev = 0;
	unsigned int it;
	for (it = 0; it < settings->max_it; it++) {
		stat->it++;
		//================== normalize matriz Z
		//scale Z
		if (settings->algorithm == L0_penalized_L1_PCA || settings->algorithm
				== L1_penalized_L1_PCA) {
			for (j = 0; j < nnz_z; j++) {
				z[j] = sgn(z[j]);
			}
		} else {
			clear_local_vector(norms, settings->starting_points);
			//data are stored in column order
			for (i = 0; i < z_nq; i++) {
				double tmp = 0;
				for (j = 0; j < z_mp; j++) {
					tmp += z[j + i * z_mp] * z[j + i * z_mp];
				}
				norms[get_column_coordinate(i, mycol, npcol, X_VECTOR_BLOCKING)]
						= tmp;
			}
			//sum up + distribute norms of "Z"
			dgsum2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER,
					&settings->starting_points, &i_one, norms,
					&settings->starting_points, &i_negone, &i_negone);
			//normalize local "z"
			for (i = 0; i < z_nq; i++) {
				double scaleNorm = 1 / sqrt(
						norms[get_column_coordinate(i, mycol, npcol,
								X_VECTOR_BLOCKING)]);
				for (j = 0; j < z_mp; j++) {
					z[j + i * z_mp] = z[j + i * z_mp] * scaleNorm;
				}
			}
		}
		//======================
		// Multiply x = B'*z
		//		sub(C) := alpha*op(sub(A))*op(sub(B)) + beta*sub(C),
		pdgemm_(&trans, &transNo, &N, &SP_MKL, &M, &one, B, &i_one, &i_one,
				descB, z, &i_one, &i_one, desc_z, &zero, V, &i_one, &i_one,
				desc_V);
		// perform thresh-holding operations and compute objective values
		clear_local_vector(norms, settings->starting_points); // we use NORMS to store objective values
		if (settings->algorithm == L0_penalized_L1_PCA || settings->algorithm
				== L0_penalized_L2_PCA) {
			for (i = 0; i < V_nq; i++) {
				for (j = 0; j < V_mp; j++) {
					double const tmp = V[j + i * V_mp];
					double tmp2 = (tmp * tmp - settings->penalty);
					if (tmp2 > 0) {
						norms[get_column_coordinate(i, mycol, npcol,
								X_VECTOR_BLOCKING)] += tmp2;
					} else {
						V[j + i * V_mp] = 0;
					}
				}
			}
		} else {
			for (i = 0; i < V_nq; i++) {
				for (j = 0; j < V_mp; j++) {
					double const tmp = V[j + i * V_mp];
					double tmp2 = myabs(tmp) - settings->penalty;
					if (tmp2 > 0) {
						norms[get_column_coordinate(i, mycol, npcol,
								X_VECTOR_BLOCKING)] += tmp2 * tmp2;
						V[j + i * V_mp] = tmp2 * sgn(tmp);
					} else {
						V[j + i * V_mp] = 0;
					}
				}
			}
		}

		//Agregate FVAL
		dgsum2d_(&ictxt, &C_CHAR_SCOPE_ALL, &C_CHAR_GENERAL_TREE_CATHER,
				&settings->starting_points, &i_one, norms,
				&settings->starting_points, &i_negone, &i_negone);
		double max_error = 0;
		for (i = 0; i < settings->starting_points; i++) {
			if (settings->algorithm == L0_penalized_L1_PCA
					|| settings->algorithm == L0_penalized_L2_PCA) {
				values[i].val = norms[i];
			} else {
				values[i].val = sqrt(norms[i]);
			}
			if (it > 0) {
				double tmp_error = computeTheError(values[i].val,
						values[i].prev_val);
				if (tmp_error > max_error)
					max_error = tmp_error;
			}
			values[i].prev_val = values[i].val;
		}
		// z= B*V
		pdgemm_(&transNo, &transNo, &M, &SP_MKL, &N, &one, B, &i_one, &i_one,
				descB, V, &i_one, &i_one, desc_V, &zero, z, &i_one, &i_one,
				desc_z);
		if (it > 0 && termination_criteria(max_error, it, settings)) { //FIXME CHECK
			break;
		}
	}
	stat->fval = -1;
	int max_selection_idx = -1;
	for (i = 0; i < settings->starting_points; i++) {
		if (values[i].val > stat->fval) {
			max_selection_idx = i;
			stat->fval = values[i].val;
		}
	}
	//copy "MAX_SELECTION_IDX" column from matrix V into vector x!
	//	sub(C):=beta*sub(C) + alpha*op(sub(A)),
	max_selection_idx++;// because next fucntion use 1-based
	pdgeadd_(&transNo, &N, &i_one, &one, V, &i_one, &max_selection_idx, desc_V,
			&zero, x, &i_one, &i_one, desc_x);
	//============== COMPUTE final "x"
	double norm_of_x = 0;
	pdnrm2_(&N, &norm_of_x, x, &i_one, &i_one, &desc_x, &i_one);
	norm_of_x = 1 / norm_of_x;
	for (i = 0; i < nnz_x; i++) {
		x[i] = x[i] * norm_of_x;
	}
	free(z);
	free(V);
	free(norms);
	free(values);
}
