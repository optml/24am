/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M. Takac and S. Damla Ahipasaoglu 
 *    "Alternating Maximization: Unified Framework and 24 Parallel Codes for L1 and L2 based Sparse PCA"
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

namespace input_ouput_helper {

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
void read_csv_file(std::vector<F> &Bmat, unsigned int &ldB, unsigned int &m,
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


void save_statistics(solver_structures::optimization_statistics* stat,
		solver_structures::optimization_settings * settings){
	ofstream stat_file;
		stat_file.open(get_file_modified_name(settings->result_file, "stat"));
		stat_file << "Solver options " << '\n';
		stat_file << "Algorithm: " << settings->algorithm << '\n';
		if (settings->isConstrainedProblem()){
			stat_file << "Constraint parameter: " << settings->constrain << '\n';
		}else{
			stat_file << "Penalty parameter: " << settings->penalty<< '\n';
		}
		stat_file << "Max iterations per starting point: " << settings->max_it<< '\n';
		stat_file << "Starting points: " << settings->starting_points<< '\n';
		stat_file << "Batch size: " << settings->batch_size<< '\n';
		stat_file << "Batching strategy (OTF): " << settings->on_the_fly_generation<< '\n';
		stat_file << "Double precision: " << settings->double_precission<< '\n';
		stat_file << "Toll: " << settings->toll<< '\n';
	#ifdef DEBUG
		stat_file << "DEBUG MODE: " << 1<< '\n';
	#endif

		stat_file << '\n'<< "Timing " << '\n';
		stat_file << "Total computational time: "<< setprecision(16) << stat->true_computation_time<< " sec"<< '\n';
		stat_file << "Total elapsed time: "<< setprecision(16) << stat->total_elapsed_time<< " sec"<<'\n';

		stat_file << '\n'<< "Result " << '\n';
		stat_file << "Objective value: " << setprecision(16)<< stat->fval<< '\n';
		stat_file << "Elapsed it (total): " << stat->it<< '\n';
		stat_file << "Average it (per starting point): "<< setprecision(16) << stat->it*settings->batch_size/(0.0+settings->starting_points)<< '\n';


		stat_file.close();
}


template<typename F>
void save_results(solver_structures::optimization_statistics* stat,
		solver_structures::optimization_settings * settings, const F* x, unsigned int lenght) {
	ofstream result_file;
	result_file.open(get_file_modified_name(settings->result_file, "x"));
	for (unsigned int i = 0; i < lenght; i++) {
		if (x[i] != 0) {
			result_file << i << "," << setprecision(16) << x[i] << '\n';
		}
	}
	result_file.close();
}

}
#endif /* FILE_READER_H_ */
