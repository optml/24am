/*
 *
 *  Created on: Mar 29, 2012
 *      Author: taki
 */

#include "timer.h"

double gettime(void) {
	struct timeval timer;
	if (gettimeofday(&timer, NULL))
		return -1.0;
	return timer.tv_sec + 1.0e-6 * timer.tv_usec;
}

