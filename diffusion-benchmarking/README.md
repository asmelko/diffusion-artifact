# diffusion-benchmarking

This repository contains various variants of parallel implementations of diffusion solver used in PhysiBoSS application. The diffusion application is designed to benchmark these implementation, validate their correctness and, also, solve the provided problems and store the outputed result.

## Build requirements

The requirements to build the application are:
- git
- C++20 compliant compiler
- CMake
- LAPACK library
- OpenMP

> We provide `.devcontainer/devcontainer.json` VSCode Devcontainer, which, after reopening the VSCode window in the Devcontainer, contains all the dependencies to build the application. 

To build `diffuse` app, write:
```bash
# Fetch dependencies
git submodule update --init --recursive
# configure cmake for release build in 'build' directory:
cmake -DCMAKE_BUILD_TYPE=Release -B build
# build the contents of 'build' directory:
cmake --build build
```

## Usage

An example usage of the app looks as follows:
```bash
./build/diffuse --problem example-problems/50x50x50x1.json --alg lstc --params example-problems/params.json --benchmark --double
```
This command benchmarks the diffusion on a problem defined in `example-problems/50x50x50x1.json` file, using algorithm named `lstc` with the parameter set defined in `example-problems/params.json` in double precision. 

Instead of providing `--benchmark` command line parameter, one can use `--validate` to check the correctness of the selected algorithm with respect to the reference algorithm `ref`. Further, `--run [FILE]` just runs the problem and outputs the result in the provided file (if selected). 

In addition, `--profile` command line parameter outputs the performance counters collected by PAPI library. You can specify the counters to collect using `PAPI_EVENTS` environment variable and also the output directory using `PAPI_OUTPUT_DIRECTORY` environment variable. To see the usable performance counters on your system, run `papi_avail` (either present on your system or built and accesible in `<BUILD DIR>/_deps/papi-build/bin` directory). The work that is being profiled is equivalent to running the application with `--run` parameter. Here is a code snippet that collects specific performance counters and outputs them into a specified output directory:

```bash
export PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC"
export PAPI_OUTPUT_DIRECTORY="scratch/measurement"
./build/diffuse --problem example-problems/50x50x50x1.json --alg biofvm --params example-problems/params.json --profile
```


## List of algorithms

The repository contains various variants of the diffusion solver, each containing either different kind of optimizations or overall completely different algorithm.

The following table contains a list of implementations that can be tested together with their short description. The full description is located in the respective header files in `src` directory.

| Name            | Full Name                                             | Description                                                                                   |
| --------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **ref**         | *Reference Solver*                                    | The baseline reference solver used for validation and correctness checks.                     |
| **lstc**        | *Least Compute*                                       | Optimized solver minimizing computational cost for Thomas algorithm.                          |
| **lstcm**       | *Least Compute Multiplied*                            | Variant of `lstc` which expands (multiplies) the vectorization along x dimension.             |
| **lstcma**      | *Least Compute Multiplied Aligned*                    | Variant of `lstcm` with aligned memory.                                                       |
| **lstct**       | *Least Compute Temporal*                              | Variant of `lstc` with improved temporal locality of data.                                    |
| **lstcta**      | *Least Compute Temporal Aligned*                      | Variant of `lstct` with aligned memory.                                                       |
| **lstcs**       | *Least Compute Substrate*                             | Variant of `lstc` where substrate dimension is the outermost.                                 |
| **lstcst**      | *Least Compute Substrate Temporal*                    | Variant of `lstcs` with improved temporal locality of data.                                   |
| **lstcsta**     | *Least Compute Substrate Temporal Aligned*            | Variant of `lstcst` with aligned memory.                                                      |
| **lstcstai**    | *Least Compute Substrate Temporal Aligned Intrinsics* | Variant of `lstcsta` which uses vector intrinsics to vectorize x dimension.                   |
| **lstm**        | *Least Memory*                                        | Solver optimized for minimal memory accesses.                                                 |
| **lstmt**       | *Least Memory Temporal*                               | Variant of `lstm` with improved temporal locality of data.                                    |
| **lstmta**      | *Least Memory Temporal Aligned*                       | Variant of `lstmt` with aligned memory.                                                       |
| **lstmtai**     | *Least Memory Temporal Aligned Intrinsics*            | Variant of `lstmta` which uses vector intrinsics to vectorize x dimension.                    |
| **lapack**      | *LAPACK Thomas Solver*                                | Solver utilizing LAPACK for positive-definite tridiagonal systems. Only supports 1D problems. |
| **lapack2**     | *General LAPACK Thomas Solver*                        | Solver utilizing LAPACK for general tridiagonal systems. Only supports 1D problems.           |
| **full_lapack** | *Full LAPACK Solver*                                  | Solver using LAPACK for positive-definite banded systems. Supports all dimensions.            |
| **avx256d**     | *SIMD Solver (AVX256 Double)*                         | SIMD-optimized solver using AVX256 intrinsics. Only for double precision.                     |
| **cr**          | *Cyclic Reduction*                                    | Solver using cyclic reduction for tridiagonal systems.                                        |
| **crt**         | *Cyclic Reduction Temporal*                           | Variant of `cr` with improved temporal locality of data.                                      |
