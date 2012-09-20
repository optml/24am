/*
//HEADER INFO
 */

/*
 * timer.h
 *
 *  Created on: Mar 29, 2012
 *      Author: taki
 */

#ifndef TIMER_H_
#define TIMER_H_
#include <time.h>
#include <sys/time.h>


double gettime(void) {
	struct timeval timer;
	if (gettimeofday(&timer, NULL))
		return -1.0;
	return timer.tv_sec + 1.0e-6 * timer.tv_usec;
}

class mytimer {
	clock_t endcl;
	clock_t startcl;
	double start_time;
	double end_time;
public:
	mytimer(){
		start_time=0;
		end_time=0;
	}
	void start() {
		startcl = clock();
		start_time = gettime();
	}
	void end() {
		endcl = clock();
		end_time = gettime();

	}

	double getElapsedWallClockTime() {
		return end_time - start_time;
	}
	double getElapsedCPUTime() {
		return ((float) endcl - (float) (startcl)) / 1000000.0F;
	}

};

#endif /* TIMER_H_ */
