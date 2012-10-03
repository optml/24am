GSL_INCLUDE = -I/exports/applications/apps/gsl/1.9/include
GSL_LIB= -L/exports/applications/apps/gsl/1.9/lib
CC = g++
BLAS_LIB= $(GSL_LIB)



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

	