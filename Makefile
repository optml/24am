CUDA_INSTALL_PATH= /exports/applications/apps/cuda/rhel5/4.2/cuda
#MKLROOT = /opt/intel/mkl
MKLROOT = /exports/applications/apps/SL5/intel/MKL/10.2.3.029
MKLLIB=$(MKLROOT)/lib/intel64
MKLINCLUDE=$(MKLROOT)/include


GSL_INCLUDE = -I/exports/applications/apps/gsl/1.9/include
GSL_LIB= -L/exports/applications/apps/gsl/1.9/lib


MPICC = mpicc
CC = g++
CUDA_COMPILER=nvcc
CUDA_INCLUDES  := -I. -I$(CUDA_INSTALL_PATH)/include
CUDA_LIB := -L./ $(BLAS_LIB) -L../objects  -lgsl -lgslcblas -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_INSTALL_PATH)/lib64  -lcublas -lm -arch sm_20
BLAS_LIB= $(GSL_LIB)


MKK_LIBS= -lmkl_scalapack_lp64  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64






UTILS = timer.o gsl_helper.o  my_cblas_wrapper.o  
OBJS = multicore_console.o  $(UTILS)
DEBUG = -g -DDEBUG
CFLAGS = -Wall -O3 -c $(DEBUG) -fopenmp 
LFLAGS = -Wall -O3 $(DEBUG) 
INCLUDE= -I. -I./frontends  -I/usr/local/include $(GSL_INCLUDE)
LIBS = -L./ $(BLAS_LIB) -L../objects -fopenmp -lgsl -lgslcblas
OBJFOL=objects/
SRC = src
UTILSFOLDER=$(SRC)/utils/
TESTFOLDER=$(SRC)/test/
GPOWERFOLDER=$(SRC)/gpower/
FRONTENDFOLDER=$(SRC)/frontends/
BUILD_FOLDER=build/
DISTR_FOLDER=$(SRC)/dgpower/

all: multicore_console test_cpu distributed_console test_multicore


distributed_time_helper.o:
	$(MPICC)  $(DEBUG) -w -u -O3 $(DISTR_FOLDER)time_helper.c -c -o $(OBJFOL)time_helper.o

distributed_termination_criteria.o:
	$(MPICC)  $(DEBUG) -w -u -O3 $(DISTR_FOLDER)termination_criteria.c -c -o $(OBJFOL)termination_criteria.o

mkl_constants_and_headers.o:
	$(MPICC)  $(DEBUG) -w -u -I. -I$(MKLROOT)/include -O3 $(DISTR_FOLDER)mkl_constants_and_headers.c -c -o $(OBJFOL)mkl_constants_and_headers.o





distributed_PCA_solver.o:
	$(MPICC)  $(DEBUG) -w -u -I. -I$(MKLROOT)/include -O3 $(DISTR_FOLDER)distributed_PCA_solver.c -c -o $(OBJFOL)distributed_PCA_solver.o

distributed_console.o:
	$(MPICC)  $(DEBUG) -w -u -O3 -I. -I$(MKLROOT)/include -I/usr/local/include -c -o  $(OBJFOL)distributed_console.o   $(FRONTENDFOLDER)distributed_console.c


distributed_console: mkl_constants_and_headers.o distributed_console.o distributed_time_helper.o distributed_termination_criteria.o   distributed_PCA_solver.o
	$(MPICC)  $(DEBUG) -o $(BUILD_FOLDER)distributed_console $(OBJFOL)distributed_console.o $(OBJFOL)termination_criteria.o $(OBJFOL)mkl_constants_and_headers.o $(OBJFOL)time_helper.o $(OBJFOL)distributed_PCA_solver.o $(BLAS_LIB) -L$(MKLLIB) -I/usr/local/include $(MKK_LIBS) -lpthread -lm -lpthread   -lgsl
 


timer.o : 
	$(CC) $(CFLAGS) $(UTILSFOLDER)timer.cpp -o $(OBJFOL)timer.o



my_cblas_wrapper.o :
	$(CC) $(CFLAGS) $(INCLUDE)  $(UTILSFOLDER)my_cblas_wrapper.cpp -o $(OBJFOL)my_cblas_wrapper.o


gsl_helper.o :
	$(CC) $(CFLAGS) $(INCLUDE) $(UTILSFOLDER)gsl_helper.cpp -o $(OBJFOL)gsl_helper.o


tests: test_pc test_distributed



multicore_console.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(INCLUDE)  $(FRONTENDFOLDER)multicore_console.cpp  -o $(OBJFOL)multicore_console.o

test_cpu.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(INCLUDE) $(TESTFOLDER)test_cpu.cpp  -o $(OBJFOL)test_cpu.o


PC_OBJECTS =    $(OBJFOL)timer.o      $(OBJFOL)my_cblas_wrapper.o  $(OBJFOL)gsl_helper.o    


gpu_console: $(OBJS)
	$(CUDA_COMPILER) -O3 $(GSL_INCLUDE)  $(CUDA_INCLUDES) $(INCLUDE)  $(FRONTENDFOLDER)gpu_console.cu   $(PC_OBJECTS) $(CUDA_LIB) $(GSL_LIB) -lgomp -o $(BUILD_FOLDER)gpu_console

gpu: gpu_console


multicore_console: $(OBJS)
	$(CC) $(LFLAGS) $(OBJFOL)multicore_console.o $(PC_OBJECTS)  $(LIBS) -o $(BUILD_FOLDER)multicore_console
    
test_cpu: test_cpu.o
	$(CC) $(LFLAGS) $(OBJFOL)test_cpu.o $(PC_OBJECTS)  $(LIBS) -o $(BUILD_FOLDER)test_cpu


test_pc:
	./$(BUILD_FOLDER)test_cpu

test_multicore:
	./$(BUILD_FOLDER)multicore_console -d datasets/small.csv  -r results/small.txt -v true -p double -a 1 -n 3
	./$(BUILD_FOLDER)multicore_console -d datasets/small.csv  -r results/small_2.txt -v true -p double -s 1000 -b 64  -a 1 -n 2
	./$(BUILD_FOLDER)multicore_console -d datasets/small.csv  -r results/small_3.txt -v true -p double -s 1000 -b 64 -u 1 -a 1 -n 2
	    
test_distributed:
	mpirun -np 8 --mca orte_base_help_aggregate 0  $(BUILD_FOLDER)distributed_console
    
    
    
clean:
	\rm $(OBJFOL)*.o  build/*   
	
build: clean	
