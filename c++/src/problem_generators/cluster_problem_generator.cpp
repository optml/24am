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

#include <iostream>
using namespace std;
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
int main() {
	const char* filename = "datasets/cluster.dat.";
	const int numOfFiles = 6;
	const int n = 100;
//	const int m =  500*1024 * 1024 / 8 / n;
	const int m = 400;

	const int ROW_GRID = 2;
	const int COL_GRID = numOfFiles / ROW_GRID;

	unsigned int seeds[numOfFiles];
	unsigned int seq[numOfFiles];
	for (int f = 0; f < numOfFiles; f++) {
		seeds[f] = f;
		seq[f] = 0;
	}

#pragma omp parallel for
	for (int f = 0; f < numOfFiles; f++) {
		printf("%d\n", f);
		char final_file[1000];
		sprintf(final_file, "%s%d-%d", filename, f % ROW_GRID, f / ROW_GRID);
		FILE * fin = fopen(final_file, "w");
		fprintf(fin, "%d;%d\n", m, n);
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
				fprintf(fin, "%f;",
						(double) (rand_r(&seeds[f]) / (0.0 + RAND_MAX)) * 1
								/ m);
			}
			fprintf(fin, "\n");
		}
		fclose(fin);
	}
	for (int f = 0; f < numOfFiles; f++) {
		seeds[f] = f;
		seq[f] = 0;
	}
	char final_file[1000];
	sprintf(final_file, "%sall", filename);
	FILE * fin = fopen(final_file, "w");
	for (int r = 0; r < ROW_GRID; r++) {
		for (int j = 0; j < m; j++) {
			for (int c = 0; c < COL_GRID; c++) {
				for (int i = 0; i < n; i++) {
					int f = r + c * ROW_GRID;
					fprintf(fin, "%f",
							(double) (rand_r(&seeds[f]) / (0.0 + RAND_MAX)) * 1
									/ m);
					if (i<n-1 || c<COL_GRID-1)
						fprintf(fin, ",");
//					fprintf(fin, "%d;",
//																seq[f]);seq[f]++;
//					fprintf(fin, "%f;",
//							f+0.0);
				}
			}
			fprintf(fin, "\n");
		}
	}
	fclose(fin);
	return 0;
}
