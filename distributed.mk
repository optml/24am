MKL_LIBS=    $(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lpthread -lm


# compiler which should be used
MPICC = mpicc
MPICPP = mpic++

#============================================================================================
# You should not modify the lines below

# CONSOLE APP
cluster_console: 
	$(MPICPP) -I$(MKLROOT)/include $(DEBUG) -o $(BUILD_FOLDER)cluster_console $(FRONTENDFOLDER)cluster_console.cpp  $(MKL_LIBS)

#PROBLEM GENERATOR
distributed_generator: 
	$(CC) -O3 -fopenmp $(DEBUG) -o $(BUILD_FOLDER)distributed_generator $(SRC)/problem_generators/cluster_problem_generator.cpp
	./$(BUILD_FOLDER)distributed_generator  
 

# DISTRIBUTED TEST
distributed_test: distributed_generator cluster_console
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/distributed.dat.  -o results/distributed.txt -v true -p double -s 1000 -b 128 -u 1 -a 5 -n 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/distributed.dat.  -o results/distributed.txt.2 -v true -p double -s 1000 -b 128 -u 1 -a 5 -n 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/distributed.dat.  -o results/distributed.txt.L0_C_L1_PCA.3 -v true -p double -s 1000 -b 128 -u 1 -a 1 -n 2 -x 2 -m 100
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/distributed.dat.  -o results/distributed.txt.L0_C_L2_PCA.4 -v true -p double -s 1000 -b 128 -u 1 -a 2 -n 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/distributed.dat.  -o results/distributed.txt.L1_C_L1_PCA.5 -v true -p double -s 10 -b 10 -u 1 -a 3 -n 2 -x 2 -m 3
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/distributed.dat.  -o results/distributed.txt.X.7 -v true -p double -s 1000 -b 128 -u 1 -a 5 -n 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/distributed.dat.  -o results/distributed.txt.X.8 -v true -p double -s 1000 -b 128 -u 1 -a 6 -n 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/distributed.dat.  -o results/distributed.txt.X.9 -v true -p double -s 1000 -b 128 -u 1 -a 7 -n 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/distributed.dat.  -o results/distributed.txt.X.10 -v true -p double -s 1000 -b 128 -u 1 -a 8 -n 2 -x 2			



distributed_unit_test: distributed_generator
	$(MPICPP) -I$(MKLROOT)/include $(DEBUG) -o $(BUILD_FOLDER)distributed_unittest $(SRC)/test/distributed_unit_test.cpp $(MKL_LIBS) -fopenmp -lgsl -lgslcblas
	mpirun  --mca orte_base_help_aggregate 0 -np 6 $(BUILD_FOLDER)distributed_unittest



 
distributed:   distributed_test distributed_unit_test



