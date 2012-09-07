/*
 * optimization_settings.h
 *
 *  Created on: Mar 26, 2012
 *      Author: taki
 */

#ifndef D_OPTIMIZATION_SETTINGS_H_
#define D_OPTIMIZATION_SETTINGS_H_

enum SparsePCA_Algorithm {
	L0_penalized_L1_PCA = 0,
	L0_penalized_L2_PCA,
	L1_penalized_L1_PCA,
	L1_penalized_L2_PCA
//	,
//	L0_constrained_L1_PCA,
//	L0_constrained_L2_PCA,
//	L1_constrained_L1_PCA,
//	L1_constrained_L2_PCA
};

struct optimization_settings {
	double toll;
	enum SparsePCA_Algorithm algorithm;
	unsigned int max_it;
	double penalty;
	unsigned int starting_points;


};

#endif /* D_OPTIMIZATION_SETTINGS_H_ */
