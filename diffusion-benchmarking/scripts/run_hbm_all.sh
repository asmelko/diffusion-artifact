#!/bin/bash

module load cmake/3.30.5
module load gcc/14.1.0_binutils241

sbatch scripts/run_mn_hbm.sh benchmarking-pb.json s1-f
sbatch scripts/run_mn_hbm.sh benchmarking-pb.json s1-d
sbatch scripts/run_mn_hbm.sh benchmarking-pb.json s2-f
sbatch scripts/run_mn_hbm.sh benchmarking-pb.json s2-d
sbatch scripts/run_mn_hbm.sh benchmarking-pb.json s4-f
sbatch scripts/run_mn_hbm.sh benchmarking-pb.json s4-d
sbatch scripts/run_mn_hbm.sh benchmarking-pb.json s8-f
sbatch scripts/run_mn_hbm.sh benchmarking-pb.json s8-d

sbatch scripts/run_mn_hbm.sh benchmarking.json s1-f
sbatch scripts/run_mn_hbm.sh benchmarking.json s1-d
sbatch scripts/run_mn_hbm.sh benchmarking.json s2-f
sbatch scripts/run_mn_hbm.sh benchmarking.json s2-d
sbatch scripts/run_mn_hbm.sh benchmarking.json s4-f
sbatch scripts/run_mn_hbm.sh benchmarking.json s4-d
sbatch scripts/run_mn_hbm.sh benchmarking.json s8-f
sbatch scripts/run_mn_hbm.sh benchmarking.json s8-d