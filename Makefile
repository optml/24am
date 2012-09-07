CUDA_INSTALL_PATH= /exports/applications/apps/cuda/rhel5/4.0/cuda
MKLROOT = /opt/intel/mkl

MPICC = mpicc
CC = g++
CUDA_INCLUDES  := -I. -I$(CUDA_INSTALL_PATH)/include
CUDA_LIB := -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_INSTALL_PATH)/lib64  -lcublas -lm -arch sm_20




MKK_LIBS= -lmkl_scalapack_core  -lmkl_intel -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi






UTILS = timer.o gsl_helper.o  my_cblas_wrapper.o  
OBJS = personal_pc.o  $(UTILS)
DEBUG = -g -DDEBUG
CFLAGS = -Wall -O3 -c $(DEBUG) -fopenmp 
LFLAGS = -Wall -O3 $(DEBUG) 
INCLUDE= -I. -I./frontends  -I/usr/local/include
LIBS = -L./ -L../objects -fopenmp -lgsl -lgslcblas
OBJFOL=objects/
SRC = src
UTILSFOLDER=$(SRC)/utils/
TESTFOLDER=$(SRC)/test/
GPOWERFOLDER=$(SRC)/gpower/
FRONTENDFOLDER=$(SRC)/frontends/
BUILD_FOLDER=build/
DISTR_FOLDER=$(SRC)/dgpower/

all: personal_pc test_cpu distributed_console


distributed_time_helper.o:
	$(MPICC)  $(DEBUG) -w -u -O3 $(DISTR_FOLDER)time_helper.c -c -o $(OBJFOL)time_helper.o

distributed_termination_criteria.o:
	$(MPICC)  $(DEBUG) -w -u -O3 $(DISTR_FOLDER)termination_criteria.c -c -o $(OBJFOL)termination_criteria.o

mkl_constants_and_headers.o:
	$(MPICC)  $(DEBUG) -w -u -I. -I$(MKLROOT)/include -O3 $(DISTR_FOLDER)mkl_constants_and_headers.c -c -o $(OBJFOL)mkl_constants_and_headers.o





distributed_PCA_solver.o:
	$(MPICC)  $(DEBUG) -w -u -I$(MKLROOT)/include -O3 $(DISTR_FOLDER)distributed_PCA_solver.c -c -o $(OBJFOL)distributed_PCA_solver.o

distributed_console.o:
	$(MPICC)  $(DEBUG) -w -u -O3 -I$(MKLROOT)/include -I/usr/local/include -c -o  $(OBJFOL)distributed_console.o   $(FRONTENDFOLDER)distributed_console.c


distributed_console: mkl_constants_and_headers.o distributed_console.o distributed_time_helper.o distributed_termination_criteria.o   distributed_PCA_solver.o
	$(MPICC)  $(DEBUG) -o $(BUILD_FOLDER)distributed_console $(OBJFOL)distributed_console.o $(OBJFOL)termination_criteria.o $(OBJFOL)mkl_constants_and_headers.o $(OBJFOL)time_helper.o $(OBJFOL)distributed_PCA_solver.o -L$(MKLROOT)/lib/ia32 -I/usr/local/include $(MKK_LIBS) -lpthread -lm -lpthread   -lgsl
 


timer.o : 
	$(CC) $(CFLAGS) $(UTILSFOLDER)timer.cpp -o $(OBJFOL)timer.o



my_cblas_wrapper.o :
	$(CC) $(CFLAGS) $(UTILSFOLDER)my_cblas_wrapper.cpp -o $(OBJFOL)my_cblas_wrapper.o


gsl_helper.o :
	$(CC) $(CFLAGS) $(UTILSFOLDER)gsl_helper.cpp -o $(OBJFOL)gsl_helper.o


tests: test_pc test_distributed



personal_pc.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(FRONTENDFOLDER)personal_pc.cpp  -o $(OBJFOL)personal_pc.o

test_cpu.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(TESTFOLDER)test_cpu.cpp  -o $(OBJFOL)test_cpu.o


PC_OBJECTS =    $(OBJFOL)timer.o      $(OBJFOL)my_cblas_wrapper.o  $(OBJFOL)gsl_helper.o    




personal_pc: $(OBJS)
	$(CC) $(LFLAGS) $(OBJFOL)personal_pc.o $(PC_OBJECTS)  $(LIBS) -o $(BUILD_FOLDER)personal_pc
    
test_cpu: test_cpu.o
	$(CC) $(LFLAGS) $(OBJFOL)test_cpu.o $(PC_OBJECTS)  $(LIBS) -o $(BUILD_FOLDER)test_cpu


test_pc:
	./$(BUILD_FOLDER)test_cpu
    
test_distributed:
	mpirun -np 8 --mca orte_base_help_aggregate 0  $(BUILD_FOLDER)distributed_console
    
    
    
clean:
	\rm $(OBJFOL)*.o  build/*   
	
build: clean	
