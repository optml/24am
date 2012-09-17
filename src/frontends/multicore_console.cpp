/*
 *  TODO
 *
 */
#include "../class/optimization_settings.h"
#include "../class/optimization_statistics.h"
using namespace solver_structures;
#include "../gpower/sparse_PCA_solver.h"
#include "../utils/file_reader.h"
#include "../utils/option_console_parser.h"

template<typename F>
void load_data_and_run_solver(optimization_settings* settings) {
	double start_wall_time = gettime();
	std::vector<F> B_mat;
	unsigned int ldB;
	unsigned int m;
	unsigned int n;
	input_ouput_helper::read_csv_file(B_mat, ldB, m, n, settings->data_file);
	optimization_statistics* stat = new optimization_statistics();
	stat->n = n;
	const F * B = &B_mat[0];

	std::vector<F> x_vec(n, 0);
	F * x = &x_vec[0];
	PCA_solver::dense_PCA_solver(B, ldB, x, m, n, settings, stat);
	double end_wall_time = gettime();
	stat->total_elapsed_time=end_wall_time-start_wall_time;
	input_ouput_helper::save_statistics_and_results(stat, settings,&x_vec[0],n);

}

int main(int argc, char *argv[]) {
	optimization_settings* settings = new optimization_settings();
	int status = parse_console_options(settings, argc, argv);
	if (status > 0)
		return status;
	if (settings->double_precission) {
		load_data_and_run_solver<double>(settings);
	} else {
		load_data_and_run_solver<float>(settings);
	}
	return 0;
}
