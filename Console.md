# Running in Console Mode #

## Examples ##

The following command will read the data from `datasets/small.csv`
and save the result into `results/small_2.txt`. It will be run in verbose mode and use double precision. The total number of starting points (= AM subroutines) is 1024; batch size is 64. The formulation which will be solved is the **L0 constrained L1 variance SPCA** and principal vector will have at most three nonzero elements.
```
./build/multicore_console -i datasets/small.csv  -o results/small.txt -v true -d double -f 2 -s 3
```

## Input data structure ##

The input format for the **Multicore** and **GPU** solvers is plain CSV file with values separated by commas and end of line separated by "\n".
Note that there is no comma at the end of line and no "\n" at the end of the last line.

Example: A data matrix with 2 rows (= samples) and 3 columns (= variables/features):
```
0.1,0.2,0.3
0.4,0.5,0.6
```

The input data for the **Cluster** solver is assumed to be distributed. Therefore, if one is going to use, for instance, 15 nodes and a 3x5 virtual grid, there have to be 15 input files. If you pass parameter `-d /datasets/distributed.dat.`
then node (0,0) will try to load the file `/datasets/distributed.dat.0-0`.
Each file contains in first line two integers which indicate how many rows and columns it contains.

Example:
```
2;3
0.1;0.2;0.3;
0.4;0.5;0.6;
```
(note that dataset is separated by ";" instead of ",").