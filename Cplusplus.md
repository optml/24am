# Embedding Sparse PCA solver into your own C++ code #

Different option than only using the console interface is directly call the Sparse PCA solver from your C++ code. In this document we will explain how it can be done.

The building for different architectures is described in [Building](Building.md) wiki page.

**Your C++ project structure** In this document we assume that structure of your code is as follows
```
/
/build
/lib
/src/main.cpp
....
```
Copy the **24am** folder into {{{/lib}} folder, hence your code will have following structure
```
/
/build
/lib/24am
/lib/24am/src
....
/src/main.cpp
....
```




### Single-core / Multi-core ###

In the top of the file `/src/main.cpp`
write
```
#include "../lib/24am/src/class/optimization_settings.h"
#include "./lib/24am/src/class/optimization_statistics.h"
using namespace SolverStructures;
#include "./lib/24am/src/gpower/sparse_PCA_solver.h"
#include "./lib/24am/src/utils/file_reader.h"
```

Then you can call the solver like
```
SPCASolver::MulticoreSolver::denseDataSolver(...);
```

### CUDA GPU ###

In the top of the file `/src/main.cu`
write
```
#include "../lib/24am/src/class/optimization_settings.h"
#include "../lib/24am/src/class/optimization_statistics.h"
#include "../lib/24am/src/utils/file_reader.h"
#include "../lib/24am/src/utils/option_console_parser.h"
#include "../lib/24am/src/gpugpower/gpu_headers.h"
```

Then you can call the solver like
```
SPCASolver::GPUSolver::denseDataSolver(...);
```

### Cluster ###

In the top of the file `/src/main.cu`
write
```
#include "../lib/24am/src/dgpower/distributed_PCA_solver.h"
#include "../lib/24am/src/utils/file_reader.h"
#include "../lib/24am/src/utils/option_console_parser.h"
```

Then you can call the solver like
```
SPCASolver::DistributedSolver::denseDataSolver(...);
```