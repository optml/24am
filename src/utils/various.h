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

#ifndef VARIOUS_H_
#define VARIOUS_H_


template<typename F>
class value_coordinate_holder {
public:
	F val;
	F prev_val;
	F tmp;
	F current_error;
	bool done;
	unsigned int idx;
	value_coordinate_holder() {
		idx=0;
		val = 0;
		tmp=0;
		done=false;
	}

	void reset(){
		val=0;
		prev_val=0;
		tmp=0;
		done=true;
	}

};

template<typename T> int sgn(T val) {
	return (val > T(0)) - (val < T(0));
}

template<typename T> T myabs(T val) {
	return (val > 0)? val:-val;
}


template<typename F>
unsigned int vector_get_nnz(const F * x,unsigned int size) {
	unsigned int nnz = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:nnz)
#endif
	for (unsigned int i = 0; i < size; i++) {
		if (x[i] != 0)
			nnz++;
	}
	return nnz;
}

template<typename F>
void clear_local_vector(F * v, const int n) {
	int i;
	for (i = 0; i < n; i++)
		v[i] = 0;
}

/*
 * this function is used only for distributed solver and returns an position of
 * starting point for given node in the grid
 */
int get_column_coordinate(const int col, const int myCol, const int numCol,
		const int blocking) {
	const int fillup = col / blocking;
	return (fillup * (numCol - 1) + myCol) * blocking + col;
}

#endif /* VARIOUS_H_ */
