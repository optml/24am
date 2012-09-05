UTILS = timer.o gsl_helper.o  my_cblas_wrapper.o  
OBJS = personal_pc.o  $(UTILS)
CC = g++
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

all: personal_pc test_cpu



timer.o : 
	$(CC) $(CFLAGS) $(UTILSFOLDER)timer.cpp -o $(OBJFOL)timer.o



my_cblas_wrapper.o :
	$(CC) $(CFLAGS) $(UTILSFOLDER)my_cblas_wrapper.cpp -o $(OBJFOL)my_cblas_wrapper.o


gsl_helper.o :
	$(CC) $(CFLAGS) $(UTILSFOLDER)gsl_helper.cpp -o $(OBJFOL)gsl_helper.o






personal_pc.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(FRONTENDFOLDER)personal_pc.cpp  -o $(OBJFOL)personal_pc.o

test_cpu.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(TESTFOLDER)test_cpu.cpp  -o $(OBJFOL)test_cpu.o


PC_OBJECTS =    $(OBJFOL)timer.o      $(OBJFOL)my_cblas_wrapper.o  $(OBJFOL)gsl_helper.o    




personal_pc: $(OBJS)
	$(CC) $(LFLAGS) $(OBJFOL)personal_pc.o $(PC_OBJECTS)  $(LIBS) -o build/personal_pc
    
test_cpu: test_cpu.o
	$(CC) $(LFLAGS) $(OBJFOL)test_cpu.o $(PC_OBJECTS)  $(LIBS) -o build/test_cpu


run_test:
	./build/test_cpu
    
    
    
    
clean:
	\rm $(OBJFOL)*.o  build/*   
	
build: clean	
