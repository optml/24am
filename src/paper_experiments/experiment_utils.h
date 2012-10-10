/*
 * experiment_utils.h
 *
 *  Created on: Oct 9, 2012
 *      Author: taki
 */

#ifndef EXPERIMENT_UTILS_H_
#define EXPERIMENT_UTILS_H_


template<typename F>
void logTime(ofstream &stream, mytimer* mt, optimization_statistics* stat,
		optimization_settings* settings, std::vector<F>& x, int m, int n) {
	int nnz = vector_get_nnz(&x[0], n);
	cout << settings->algorithm << "," << nnz << "," << m << "," << n << ","
			<< mt->getElapsedWallClockTime() << ","
			<< stat->true_computation_time << "," << settings->batch_size << ","
			<< settings->on_the_fly_generation
			<< ","<<stat->total_threads_used
			<< ","<<settings->starting_points
			<< ","<<stat->it
			<< endl;
	stream<< settings->algorithm << "," << nnz << "," << m << "," << n << ","
			<< mt->getElapsedWallClockTime() << ","
			<< stat->true_computation_time << "," << settings->batch_size << ","
			<< settings->on_the_fly_generation
			<< ","<<stat->total_threads_used
			<< ","<<settings->starting_points
			<< ","<<stat->it
			<< endl;
}



#endif /* EXPERIMENT_UTILS_H_ */
