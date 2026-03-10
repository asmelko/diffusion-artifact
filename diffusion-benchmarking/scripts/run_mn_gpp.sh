#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --qos=gp_resa
#SBATCH --account=cns119
# #SBATCH --qos=gp_bscls
# #SBATCH --account=bsc08
#SBATCH -t 3-00:00:00
#SBATCH --exclusive
# #SBATCH --constraint=perfparanoid 


cd benchmarking
export PYTHONUNBUFFERED=1
python3 benchmark.py $1 ../build/gpp/diffuse --prefix gpp- -g $2
