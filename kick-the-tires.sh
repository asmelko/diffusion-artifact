#!/bin/bash

cd diffusion-benchmarking
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)

mkdir -p ../results
cd ../results
python3 ../diffusion-benchmarking/benchmarking/benchmark.py ../diffusion-benchmarking/benchmarking/benchmarking-fast.json ../diffusion-benchmarking/build/diffuse --prefix fast- -g s1-f
python3 ../diffusion-benchmarking/benchmarking/benchmark.py ../diffusion-benchmarking/benchmarking/benchmarking-fast.json ../diffusion-benchmarking/build/diffuse --prefix fast- -g s1-d
python3 ../diffusion-benchmarking/benchmarking/benchmark.py ../diffusion-benchmarking/benchmarking/benchmarking-fast.json ../diffusion-benchmarking/build/diffuse --prefix fast- --merge

cd ../plots
Rscript plot-fast.R
