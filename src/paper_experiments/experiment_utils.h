/*
 * experiment_utils.h
 *
 *  Created on: Oct 9, 2012
 *      Author: taki
 */

#ifndef EXPERIMENT_UTILS_H_
#define EXPERIMENT_UTILS_H_


template<typename F>
void logTime(ofstream &stream, mytimer* mt, optimization_Statisticsistics* optimizationStatistics,
		optimization_settings* optimizationSettings, std::vector<F>& x, int m, int n) {
	int nnz = vector_get_nnz(&x[0], n);
	cout << optimizationSettings->algorithm << "," << nnz << "," << m << "," << n << ","
			<< mt->getElapsedWallClockTime() << ","
			<< optimizationStatistics->true_computation_time << "," << optimizationSettings->batch_size << ","
			<< optimizationSettings->on_the_fly_generation
			<< ","<<optimizationStatistics->total_threads_used
			<< ","<<optimizationSettings->starting_points
			<< ","<<optimizationStatistics->it
			<< endl;
	stream<< optimizationSettings->algorithm << "," << nnz << "," << m << "," << n << ","
			<< mt->getElapsedWallClockTime() << ","
			<< optimizationStatistics->true_computation_time << "," << optimizationSettings->batch_size << ","
			<< optimizationSettings->on_the_fly_generation
			<< ","<<optimizationStatistics->total_threads_used
			<< ","<<optimizationSettings->starting_points
			<< ","<<optimizationStatistics->it
			<< endl;
}



#endif /* EXPERIMENT_UTILS_H_ */
