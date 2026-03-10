#!/bin/bash

# Define the list of algorithms to test
#algorithms=("ref" "lstc" "lstcm" "lstcma" "lstct" "lstcta" "lstcs" "lstcst" "lstcsta" "lstm" "lstmt" "lstmta" "avx256d" "biofvm" "lapack" "lapack2" "full_lapack")
algorithms=("ref" "lstc" "lstcm" "lstcma" "lstct" "lstcta" "lstcs" "lstcst" "lstcsta" "lstcstai" "lstm" "lstmt" "lstmta" "lstmtai" "avx256d" "biofvm" "lapack" "lapack2" "full_lapack" "cr" "crt" "sblocked" "blocked" "blockedt" "blockedta" "cubed")
# Define the common command parameters
problem_file="example-problems/300x300x300x100.json"

# Loop through each algorithm and run the command
for alg in "${algorithms[@]}"; do
    echo "Running single precision benchmark for algorithm: $alg"
    build/diffuse --problem "$problem_file" --alg "$alg" --validate
    echo "Running double precision benchmark for algorithm: $alg"
    build/diffuse --problem "$problem_file" --alg "$alg" --validate --double
    echo "-----------------------------------"
done