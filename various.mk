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

LIBS_GSL = -lgsl  -lgslcblas
LIBS_MKL =   -lgsl
