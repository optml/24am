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
	./$(BUILD_FOLDER)multicore_console -i datasets/small.csv  -o results/small.txt -v true -p double -a 1 -n 3
	./$(BUILD_FOLDER)multicore_console -i datasets/small.csv  -o results/small_2.txt -v true -p double -s 1000 -b 64  -a 1 -n 2
	./$(BUILD_FOLDER)multicore_console -i datasets/small.csv  -o results/small_3.txt -v true -p double -s 1000 -b 64 -u 1 -a 1 -n 2

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

multicore_paper_experiments_boxplot: KMP	
	$(CC) $(CFLAGS) $(INCLUDE)  $(EXPERIMENTS_FOLDER)experiment_boxplot.cpp  -o $(OBJFOL)experiment_boxplot.o 
	$(CC) $(LFLAGS) $(OBJFOL)experiment_boxplot.o  $(LIBS) -o $(BUILD_FOLDER)experiment_boxplot
	./$(BUILD_FOLDER)experiment_boxplot	


multicore_paper_experiments_text_corpora: KMP	
	$(CC) $(CFLAGS) $(INCLUDE) -I$(MKLROOT)/include $(EXPERIMENTS_FOLDER)experiment_text_corpora.cpp  -o $(OBJFOL)experiment_text_corpora.o 
	$(CC) $(LFLAGS) $(OBJFOL)experiment_text_corpora.o  $(LIBS) -o $(BUILD_FOLDER)experiment_text_corpora
	./$(BUILD_FOLDER)experiment_text_corpora	



	
multicore_paper_experiments: KMP multicore_paper_experiments_speedup multicore_paper_experiments_otf multicore_paper_experiments_batching	 





KMP:
	export KMP_AFFINITY=verbose,granularity=fine,scatter	

	
	