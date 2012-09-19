



UTILSFOLDER=$(SRC)/utils/
TESTFOLDER=$(SRC)/test/
GPOWERFOLDER=$(SRC)/gpower/
FRONTENDFOLDER=$(SRC)/frontends/
BUILD_FOLDER=build/
DISTR_FOLDER=$(SRC)/dgpower/

DEBUG = -g -DDEBUG
CFLAGS = -Wall -w -O3 -c $(DEBUG) -fopenmp 
LFLAGS = -Wall -w -O3 $(DEBUG) 
INCLUDE= -I. -I./frontends  -I/usr/local/include $(GSL_INCLUDE)
LIBS = -L./ $(BLAS_LIB) -L../objects -fopenmp -lgsl -lgslcblas
OBJFOL=objects/
SRC = src



include multicore.mk
include distributed.mk
include gpu.mk


    
    
clean:
	\rm $(OBJFOL)*.o  build/*   
	
build: clean	
