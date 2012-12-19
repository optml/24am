# Path to GSL library
GSL_INCLUDE = -I/exports/applications/apps/gsl/1.9/include
GSL_LIB= -L/exports/applications/apps/gsl/1.9/lib


# Path to Intel MKL + Linking settings
MKLROOT = /exports/applications/apps/SL5/intel/MKL/10.2.3.029
MKL_MULTICORE_LIB =    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lgsl -lm  
#MKL_MULTICORE_LIB =   -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -ldl -lpthread -lm -mkl=parallel 
MKLROOT = /home/taki/Programs/intel/mkl


# for INTEL COMPILER
#CC = icc
#OPENMP_FLAG=-openmp
# for GCC COMPILER
CC = g++
OPENMP_FLAG=-fopenmp

include various.mk

#Choose CBLAS interface GSL/MKL
#LIBS_BLAS= $(LIBS_GSL)
#BLAS_LIB= $(GSL_LIB)
LIBS_BLAS= $(LIBS_MKL)
BLAS_LIB= $(MKL_MULTICORE_LIB)



LIBS = -L./ $(BLAS_LIB) -L../objects $(OPENMP_FLAG)   $(LIBS_BLAS)
include multicore.mk
include distributed.mk
include gpu.mk


    
    
clean:
	\rm $(OBJFOL)*.o  build/*   
	
build: clean	
