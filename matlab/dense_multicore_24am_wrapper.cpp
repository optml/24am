#include <matrix.h>
#include <mex.h>   

#include "../src/class/optimization_settings.h"
#include "../src/class/optimization_statistics.h"
using namespace SolverStructures;
#include "../src/gpower/sparse_PCA_solver.h"

/* Definitions to keep compatibility with earlier versions of ML */
#ifndef MWSIZE_MAX
typedef int mwSize;
typedef int mwIndex;
typedef int mwSignedIndex;

#if (defined(_LP64) || defined(_WIN64)) && !defined(MX_COMPAT_32)
/* Currently 2^48 based on hardware limitations */
# define MWSIZE_MAX    281474976710655UL
# define MWINDEX_MAX   281474976710655UL
# define MWSINDEX_MAX  281474976710655L
# define MWSINDEX_MIN -281474976710655L
#else
# define MWSIZE_MAX    2147483647UL
# define MWINDEX_MAX   2147483647UL
# define MWSINDEX_MAX  2147483647L
# define MWSINDEX_MIN -2147483647L
#endif
#define MWSIZE_MIN    0UL
#define MWINDEX_MIN   0UL
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

//declare variables
	mxArray *B_mat, *params, *x_out_n;
	const mwSize *dims;
	double *x, *B, *paramsPtr;
	int dim_m, dim_n;
	int i, j;

//associate inputs
	B_mat = mxDuplicateArray(prhs[0]);
	params = mxDuplicateArray(prhs[1]);
//figure out dimensions
	dims = mxGetDimensions(prhs[0]);
	dim_m = (int) dims[0];
	dim_n = (int) dims[1];

//associate outputs
	x_out_n = plhs[0] = mxCreateDoubleMatrix(dim_n, 1, mxREAL);

//associate pointers
	B = mxGetPr(B_mat);
	x = mxGetPr(x_out_n);
	paramsPtr = mxGetPr(params);

// set parameters
	OptimizationStatistics* optimizationStatistics =
			new OptimizationStatistics();
	optimizationStatistics->n = dim_n;

	OptimizationSettings* optimizationSettings = new OptimizationSettings();
	optimizationSettings->constraintParameter = dim_n;
	switch ((int) paramsPtr[0]) {
	case 0:
		optimizationSettings->formulation = L0_constrained_L2_PCA;
		break;
	case 1:
		optimizationSettings->formulation = L0_constrained_L1_PCA;
		break;
	case 2:
		optimizationSettings->formulation = L1_constrained_L2_PCA;
		break;
	case 3:
		optimizationSettings->formulation = L1_constrained_L1_PCA;
		break;
	case 4:
		optimizationSettings->formulation = L0_penalized_L2_PCA;
		break;
	case 5:
		optimizationSettings->formulation = L0_penalized_L1_PCA;
		break;
	case 6:
		optimizationSettings->formulation = L1_penalized_L2_PCA;
		break;
	case 7:
		optimizationSettings->formulation = L1_penalized_L1_PCA;
		break;

	}

	optimizationSettings->constraintParameter = (int) paramsPtr[1];
	optimizationSettings->penaltyParameter = paramsPtr[1];
	optimizationSettings->tolerance = paramsPtr[2];
	optimizationSettings->maximumIterations = paramsPtr[3];
	optimizationSettings->totalStartingPoints = paramsPtr[4];
	optimizationSettings->batchSize = paramsPtr[5];
	optimizationSettings->useDoublePrecision = true;
	optimizationSettings->useOTF = true;

// run SOLVER
	SPCASolver::MulticoreSolver::denseDataSolver(B, dim_m, x, dim_m, dim_n,
			optimizationSettings, optimizationStatistics);

	return;
}
