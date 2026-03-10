#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
# #SBATCH --qos=gp_resa
# #SBATCH --account=cns119
# #SBATCH --qos=gp_bscls
# #SBATCH --account=bsc08
#SBATCH -t 1-00:00:00
#SBATCH --exclusive
#SBATCH --constraint=perfparanoid 

script=$1
build_dir=$2
out_dir=$3


cmake -S . -B ${build_dir} -DCMAKE_BUILD_TYPE=Release
cmake --build ${build_dir} --parallel 112

${script} ${build_dir}/diffuse ${out_dir} 
