# Artifact Submission: NUMA-aware diffusion solver 

This is a replication package containing code and experimental results related to the paper titled: HIGH-PERFORMANCE NUMA-AWARE IMPLEMENTATION FOR1
THE DIFFUSION EQUATION.

## Overview

The artifact comprises the following directories:

* `benchmark` -- benchmarking scripts
* `data` -- the (`bnd`, `cfg`) model pairs on which the benchmarks were run
* `plots` -- scripts for generating results plots	
* `presented-results` -- plots (including some that were not included in the paper), CSV data files with measurements and R script that generated the plots from the data
* `diffusion-benchmarking` -- all the implementations of diffusion solvers


## Detailed artifact contents

`diffusion-benchmarking/src` directory contains the source files to the algorithm kernels. Notably, `*_solver.h` files each contain a single implementation of the diffusion solver. See the header comments for details on the implementation. 


## Requirements for running the experiments

Software requirements:

* `cmake 3.22` or later 
* `R` software for plotting the graphs (see details below)
* `python3` for running the benchmarking scripts

Let us present a few commands for your convenience that will allow you to set up the environment quickly:

Installing all dependencies on Debian/Ubuntu:
```
sudo apt-get update && sudo apt-get install -y g++ cmake r-base python3
```

Installing all dependencies on RHEL-like distribution:
```
sudo dnf install -y cmake gcc-c++ R python3
```

R packages necessary for generating the plots:
```
R -e "install.packages(c('ggplot2', 'cowplot', 'sitools', 'viridis', 'dplyr', 'tidyr'), repos='https://cloud.r-project.org')"
```


## Running the experiments

Our experiments are designed to provide a comprehensive analysis of the aforementioned algorithms running various combinations of parameters computing different sizes of input instances. Therefore, the overall duration of **running the experiments is quite long** (around **2 to 3 days** on MareNostrum SC).

To provide a swift way to check the reproducibility of our experiments, we prepared a special script that runs only a subset of the benchmarks.

**Kick the tires:**

Just to see whether the code is working, run the following from the root directory:
```
./kick-the-tires.sh
```
The script should take just a few minutes to finish. The script runs a subset of the experiments.

After the script runs, it will generate results in a CSV format in the `results` directory. It should contain 2 CSV files for space-local and temporary-local algorithms respectively. Each CSV file contains self-documenting headers. Finally, the plotting script is executed generating a single plot in the `plots` directory. 
More details on how the CSV results rows are processed into plots can be found in the `plots/plots-fast.R` script.

The generated plot file will be named `data-local-normalized.pdf` and it shows the comparison of the aforementioned algorithms.


**Complete set of measurements:**

To run the complete benchmark, execute
```
./run-all.sh
```

## Measured results

The measured data and plots were stored in the `presented-results` directory. The directory also contains `plots.R` script, which was used to plot the data. It can be executed by `Rscript plots.R` if you wish to re-generate the plots from the data yourself.