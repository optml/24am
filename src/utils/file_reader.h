/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M. Takac and S. Damla Ahipasaoglu 
 *    "Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 * 
 */

#ifndef FILE_READER_H_
#define FILE_READER_H_

#include <vector>
#include<iostream>
#include<fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>
using namespace std;

namespace InputOuputHelper {

void parse_data_size_from_CSV_file(unsigned int &m, unsigned int &n,
		const char* input_csv_file) {
	m = 0;
	n = 0;
	std::ifstream data(input_csv_file);
	std::string line;
	while (std::getline(data, line)) {
		if (m == 0) {
			std::stringstream lineStream(line);
			std::string cell;
			while (std::getline(lineStream, cell, ',')) {
				n++;
			}
		}
		m++;
	}
}

/**
 * Parses data from file
 *
 * @param m - number of rows of matrix stored in CSV file
 * @param n - number of columns of matrix stored in CSV file
 * @param filename - filename of file to be read from
 */
template<typename D>
int parse_data_from_CSV_file(unsigned int m, unsigned int n,
		const char* input_csv_file, std::vector<D> & data) {
	data.resize(n * m);
	ifstream my_read_file;
	my_read_file.open(input_csv_file);
	D value;
	if (my_read_file.is_open()) {
		for (unsigned int row = 0; row < m; row++) {
			for (unsigned int col = 0; col < n; col++) {
				my_read_file >> value;
				char c;
				if (col < n - 1)
					my_read_file >> c;
				data[row + col * m] = value;
			}
		}
		my_read_file.close();
		return 0;
	} else {
		return 1;
	}
}

template<typename F>
void readCSVFile(std::vector<F> &Bmat, unsigned int &ldB, unsigned int &m,
		unsigned int & n, const char* input_csv_file) {
	parse_data_size_from_CSV_file(m, n, input_csv_file);
	parse_data_from_CSV_file(m, n, input_csv_file, Bmat);
	ldB = m;
}

char* get_file_modified_name(const char* base, string suffix) {
	stringstream ss;
	string finalFileName = base;
	ss << finalFileName;
	ss << "_" << suffix;
	finalFileName = ss.str();
	char* cstr = new char[finalFileName.size() + 1];
	strcpy(cstr, finalFileName.c_str());
	return cstr;
}


void save_optimizationStatisticsistics(SolverStructures::OptimizationStatisticsistics* optimizationStatistics,
		SolverStructures::OptimizationSettings * optimizationSettings){
	ofstream optimizationStatistics_file;
		optimizationStatistics_file.open(get_file_modified_name(optimizationSettings->result_file, "optimizationStatistics"));
		optimizationStatistics_file << "Solver options " << '\n';
		optimizationStatistics_file << "Algorithm: " << optimizationSettings->algorithm << '\n';
		if (optimizationSettings->isConstrainedProblem()){
			optimizationStatistics_file << "Constraint parameter: " << optimizationSettings->constrain << '\n';
		}else{
			optimizationStatistics_file << "Penalty parameter: " << optimizationSettings->penalty<< '\n';
		}
		optimizationStatistics_file << "Max iterations per starting point: " << optimizationSettings->max_it<< '\n';
		optimizationStatistics_file << "Starting points: " << optimizationSettings->starting_points<< '\n';
		optimizationStatistics_file << "Batch size: " << optimizationSettings->batch_size<< '\n';
		optimizationStatistics_file << "Batching strategy (OTF): " << optimizationSettings->onTheFlyMethod<< '\n';
		optimizationStatistics_file << "Double precision: " << optimizationSettings->double_precission<< '\n';
		optimizationStatistics_file << "Toll: " << optimizationSettings->toll<< '\n';
	#ifdef DEBUG
		optimizationStatistics_file << "DEBUG MODE: " << 1<< '\n';
	#endif

		optimizationStatistics_file << '\n'<< "Timing " << '\n';
		optimizationStatistics_file << "Total computational time: "<< setprecision(16) << optimizationStatistics->true_computation_time<< " sec"<< '\n';
		optimizationStatistics_file << "Total elapsed time: "<< setprecision(16) << optimizationStatistics->total_elapsed_time<< " sec"<<'\n';

		optimizationStatistics_file << '\n'<< "Result " << '\n';
		optimizationStatistics_file << "Objective value: " << setprecision(16)<< optimizationStatistics->fval<< '\n';
		optimizationStatistics_file << "Elapsed it (total): " << optimizationStatistics->it<< '\n';
		optimizationStatistics_file << "Average it (per starting point): "<< setprecision(16) << optimizationStatistics->it*optimizationSettings->batch_size/(0.0+optimizationSettings->starting_points)<< '\n';


		optimizationStatistics_file.close();
}


template<typename F>
void save_results(SolverStructures::OptimizationStatisticsistics* optimizationStatistics,
		SolverStructures::OptimizationSettings * optimizationSettings, const F* x, unsigned int lenght) {
	ofstream result_file;
	result_file.open(get_file_modified_name(optimizationSettings->result_file, "x"));
	for (unsigned int i = 0; i < lenght; i++) {
		if (x[i] != 0) {
			result_file << i << "," << setprecision(16) << x[i] << '\n';
		}
	}
	result_file.close();
}

}
#endif /* FILE_READER_H_ */
