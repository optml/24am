#MKLROOT = /home/taki/Programs/intel/mkl
MKLROOT = /exports/applications/apps/SL5/intel/MKL/10.2.3.029

# for INTEL COMPILER
CC = icc
OPENMP_FLAG=-openmp
# for GCC COMPILER
#CC = g++
#OPENMP_FLAG=-fopenmp


UTILSFOLDER=$(SRC)/utils/
TESTFOLDER=$(SRC)/test/
GPOWERFOLDER=$(SRC)/gpower/
FRONTENDFOLDER=$(SRC)/frontends/
BUILD_FOLDER=build/
DISTR_FOLDER=$(SRC)/dgpower/
EXPERIMENTS_FOLDER=$(SRC)/paper_experiments/


DEBUG = -g -DDEBUG
CFLAGS = -Wall -w -O3 -c $(DEBUG) $(OPENMP_FLAG) 
LFLAGS = -Wall -w -O3 $(DEBUG)  $(OPENMP_FLAG)
INCLUDE= -I. -I./frontends  -I/usr/local/include $(GSL_INCLUDE)
OBJFOL=objects/
SRC = src

GSL_INCLUDE = -I/exports/applications/apps/gsl/1.9/include
GSL_LIB= -L/exports/applications/apps/gsl/1.9/lib

#MKL_MULTICORE_LIB =   -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -ldl -lpthread -lm -mkl=parallel 
MKL_MULTICORE_LIB =    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lgsl -lm  


BLAS_LIB= $(MKL_MULTICORE_LIB)
LIBS_GSL = -lgsl  -lgslcblas
LIBS_MKL =   -lgsl
#LIBS_BLAS= $(LIBS_GSL)
LIBS_BLAS= $(LIBS_MKL)
#BLAS_LIB= $(GSL_LIB) 

LIBS = -L./ $(BLAS_LIB) -L../objects $(OPENMP_FLAG)   $(LIBS_GSL)



include multicore.mk
include distributed.mk
include gpu.mk


    
    
clean:
	\rm $(OBJFOL)*.o  build/*   
	
build: clean	
