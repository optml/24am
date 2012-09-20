MKLROOT = /home/taki/Programs/intel/mkl
#MKLROOT = /exports/applications/apps/SL5/intel/MKL/10.2.3.029
MKL_LIBS=    $(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lpthread -lm


# compiler which should be used
MPICC = mpicc
MPICPP = mpic++

#============================================================================================
# You should not modify the lines below

# CONSOLE APP
distributed_console: 
	$(MPICPP) -I$(MKLROOT)/include $(DEBUG) -o $(BUILD_FOLDER)distributed_console $(FRONTENDFOLDER)distributed_console.cpp  $(MKL_LIBS)

#PROBLEM GENERATOR
distributed_generator: 
	$(CC) -O3 -fopenmp $(DEBUG) -o $(BUILD_FOLDER)distributed_generator $(SRC)/problem_generators/cluster_problem_generator.cpp
	./$(BUILD_FOLDER)distributed_generator  
 

# DISTRIBUTED TEST
distributed_test: distributed_generator distributed_console
	 mpirun -np 4 build/distributed_console -d datasets/distributed.dat.  -r results/distributed.txt -v true -p double -s 1000 -b 64 -u 1 -a 1 -n 2


 
distributed: distributed_test



