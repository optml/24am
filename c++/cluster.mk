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
cluster_generator: 
	$(CC) -O3 -fopenmp $(DEBUG) -o $(BUILD_FOLDER)cluster_generator $(SRC)/problem_generators/cluster_problem_generator.cpp
	./$(BUILD_FOLDER)cluster_generator  
 

# DISTRIBUTED TEST
cluster_test: cluster_generator cluster_console
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/cluster.dat.  -o results/cluster.txt -v true -d double -l 1000 -r 128 -u 1 -f 5 -s 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/cluster.dat.  -o results/cluster.txt.2 -v true -d double -l 1000 -r 128 -u 1 -f 5 -s 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/cluster.dat.  -o results/cluster.txt.L0_C_L1_PCA.3 -v true -d double -l 1000 -r 128 -u 1 -f 1 -s 2 -x 2 -m 100
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/cluster.dat.  -o results/cluster.txt.L0_C_L2_PCA.4 -v true -d double -l 1000 -r 128 -u 1 -f 2 -s 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/cluster.dat.  -o results/cluster.txt.L1_C_L1_PCA.5 -v true -d double -l 10 -r 10 -u 1 -f 3 -s 2 -x 2 -m 3
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/cluster.dat.  -o results/cluster.txt.X.7 -v true -d double -l 1000 -r 128 -u 1 -f 5 -s 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/cluster.dat.  -o results/cluster.txt.X.8 -v true -d double -l 1000 -r 128 -u 1 -f 6 -s 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/cluster.dat.  -o results/cluster.txt.X.9 -v true -d double -l 1000 -r 128 -u 1 -f 7 -s 2 -x 2
	mpirun  --mca orte_base_help_aggregate 0 -np 6 build/cluster_console -i datasets/cluster.dat.  -o results/cluster.txt.X.10 -v true -d double -l 1000 -r 128 -u 1 -f 8 -s 2 -x 2			



cluster_unit_test: cluster_generator
	$(MPICPP) -I$(MKLROOT)/include $(DEBUG) -o $(BUILD_FOLDER)cluster_unittest $(SRC)/test/cluster_unit_test.cpp $(MKL_LIBS) -fopenmp -lgsl -lgslcblas
	mpirun  --mca orte_base_help_aggregate 0 -np 6 $(BUILD_FOLDER)cluster_unittest



 
cluster:   cluster_test cluster_unit_test



