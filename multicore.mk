GSL_INCLUDE = -I/exports/applications/apps/gsl/1.9/include
GSL_LIB= -L/exports/applications/apps/gsl/1.9/lib
CC = icc
#MKL_MULTICORE_LIB =   -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -ldl -lpthread -lm -mkl=parallel 
MKL_MULTICORE_LIB =    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lgsl -lm  

#BLAS_LIB= $(GSL_LIB)
BLAS_LIB= $(MKL_MULTICORE_LIB)
LIBS_GSL = -lgslcblas
LIBS_MKL =  


LIBS = -L./ $(BLAS_LIB) -L../objects $(OPENMP_FLAG)  -lgsl $(LIBS_MKL)


OBJS = multicore_console.o  


multicore_console.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(INCLUDE)  $(FRONTENDFOLDER)multicore_console.cpp  -o $(OBJFOL)multicore_console.o
test_cpu.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(INCLUDE) $(TESTFOLDER)test_cpu.cpp  -o $(OBJFOL)test_cpu.o
multicore_console: $(OBJS)
	$(CC) $(LFLAGS) $(OBJFOL)multicore_console.o  $(LIBS) -o $(BUILD_FOLDER)multicore_console

test_cpu: test_cpu.o
	$(CC) $(LFLAGS) $(OBJFOL)test_cpu.o   $(LIBS) -o $(BUILD_FOLDER)test_cpu


test_pc:
	./$(BUILD_FOLDER)test_cpu

test_multicore:
	./$(BUILD_FOLDER)multicore_console -d datasets/small.csv  -r results/small.txt -v true -p double -a 1 -n 3
	./$(BUILD_FOLDER)multicore_console -d datasets/small.csv  -r results/small_2.txt -v true -p double -s 1000 -b 64  -a 1 -n 2
	./$(BUILD_FOLDER)multicore_console -d datasets/small.csv  -r results/small_3.txt -v true -p double -s 1000 -b 64 -u 1 -a 1 -n 2

multicore: multicore_console test_multicore


multicore_paper_experiments_batching: KMP
	$(CC) $(CFLAGS) $(INCLUDE)  $(EXPERIMENTS_FOLDER)experiment_batching.cpp  -o $(OBJFOL)experiment_batching.o 
	$(CC) $(LFLAGS) $(OBJFOL)experiment_batching.o  $(LIBS) -o $(BUILD_FOLDER)experiment_batching
	./$(BUILD_FOLDER)experiment_batching
	
multicore_paper_experiments_otf: KMP
	$(CC) $(CFLAGS) $(INCLUDE)  $(EXPERIMENTS_FOLDER)experiment_otf.cpp  -o $(OBJFOL)experiment_otf.o 
	$(CC) $(LFLAGS) $(OBJFOL)experiment_otf.o  $(LIBS) -o $(BUILD_FOLDER)experiment_otf
	./$(BUILD_FOLDER)experiment_otf
	
	
multicore_paper_experiments_speedup: KMP	
	$(CC) $(CFLAGS) $(INCLUDE)  $(EXPERIMENTS_FOLDER)experiment_multicore_speedup.cpp  -o $(OBJFOL)experiment_multicore_speedup.o 
	$(CC) $(LFLAGS) $(OBJFOL)experiment_multicore_speedup.o  $(LIBS) -o $(BUILD_FOLDER)experiment_multicore_speedup
	./$(BUILD_FOLDER)experiment_multicore_speedup	
	
multicore_paper_experiments: KMP multicore_paper_experiments_speedup multicore_paper_experiments_otf multicore_paper_experiments_batching	 


KMP:
	export KMP_AFFINITY=verbose,granularity=fine,scatter	

test:
	icc -openmp -Wall -O3 -I/exports/applications/apps/SL5/intel/MKL/10.2.3.029/include -I/exports/applications/apps/gsl/1.9/include  -c $(EXPERIMENTS_FOLDER)experiment_multicore_speedup.cpp -o $(OBJFOL)experiment_multicore_speedup.o 
	icc -O3 $(OBJFOL)experiment_multicore_speedup.o	 -o $(BUILD_FOLDER)experiment_multicore_speedup -L/exports/applications/apps/gsl/1.9/lib -L/exports/applications/apps/SL5/intel/MKL/10.2.3.029/lib/em64t  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lgsl -lm  
	./$(BUILD_FOLDER)experiment_multicore_speedup	 	
	
	