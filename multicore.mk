GSL_INCLUDE = -I/exports/applications/apps/gsl/1.9/include
GSL_LIB= -L/exports/applications/apps/gsl/1.9/lib
CC = g++
BLAS_LIB= $(GSL_LIB)



UTILS =  gsl_helper.o  my_cblas_wrapper.o  
OBJS = multicore_console.o  $(UTILS)


my_cblas_wrapper.o :
	$(CC) $(CFLAGS) $(INCLUDE)  $(UTILSFOLDER)my_cblas_wrapper.cpp -o $(OBJFOL)my_cblas_wrapper.o
gsl_helper.o :
	$(CC) $(CFLAGS) $(INCLUDE) $(UTILSFOLDER)gsl_helper.cpp -o $(OBJFOL)gsl_helper.o
multicore_console.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(INCLUDE)  $(FRONTENDFOLDER)multicore_console.cpp  -o $(OBJFOL)multicore_console.o
test_cpu.o :  $(UTILS) 
	$(CC) $(CFLAGS) $(INCLUDE) $(TESTFOLDER)test_cpu.cpp  -o $(OBJFOL)test_cpu.o
PC_OBJECTS =          $(OBJFOL)my_cblas_wrapper.o  $(OBJFOL)gsl_helper.o    
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

multicore: multicore_console test_multicore

	