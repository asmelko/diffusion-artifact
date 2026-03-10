#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --qos=gp_resa
#SBATCH --account=cns119
# #SBATCH --qos=gp_bscls
# #SBATCH --account=bsc08
#SBATCH -t 1-00:00:00
#SBATCH --exclusive
# #SBATCH --constraint=perfparanoid 

module load cmake/3.30.5
module load gcc/14.1.0_binutils241


cmake -S . -B build/gpp -DCMAKE_CXX_FLAGS="-Ofast -DNDEBUG"
cmake --build build/gpp --parallel 112