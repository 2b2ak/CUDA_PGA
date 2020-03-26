

#ifndef __ALL_BENCHMARKS_CUH__
#define __ALL_BENCHMARKS_CUH__
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_string.h>    // helper for string parsing
#include <helper_image.h>     // helper for image and data comparison
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include "device_launch_parameters.h"

#include "RuleInitializer.cuh"
#include "Fitness.cuh"
#include "Crossover.cuh"
#include "Mutation.cuh"
#include "Test.cuh"

#endif