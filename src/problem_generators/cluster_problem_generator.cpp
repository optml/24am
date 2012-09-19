//============================================================================
// Name        : GPower.cpp
// Author      : Martin Takac
// Version     :
// Copyright   : GNU
// Description : GPower Method
//============================================================================

#include <iostream>
using namespace std;
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
int main() {
	const char* filename = "datasets/distributed.dat.";
//			"/exports/work/maths_oro/taki/generated.dat.";
//	"/exports/work/scratch/taki/generated.dat.";
//	"/document/phd/c/GPower/resources/data.txt";
	const int numOfFiles = 4;
	const int n = 100;
//	const int m =  500*1024 * 1024 / 8 / n;
	const int m= 600;

	const int ROW_GRID=2;
	const int COL_GRID=numOfFiles/ROW_GRID;


#pragma omp parallel for
	for (int f = 0; f < numOfFiles; f++) {
		printf("%d\n",f);
		char final_file[1000];
		unsigned int seed=f;
		sprintf(final_file, "%s%d-%d", filename, f%ROW_GRID,f/ROW_GRID);
		FILE * fin = fopen(final_file, "w");
		fprintf(fin,"%d;%d\n", m, n);
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
              fprintf(fin,"%f;", (double)  rand_r(&seed)/RAND_MAX*1/m);
			}
			fprintf(fin,"\n");
		}

		fclose(fin);

	}
	return 0;
}
