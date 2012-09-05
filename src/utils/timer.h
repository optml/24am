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

double gettime(void);

class mytimer {
	clock_t endcl;
	clock_t startcl;
	double start_time;
	double end_time;
public:
	mytimer(){
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
