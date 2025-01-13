#!/bin/sh

#SBATCH --job-name=X0
#SBATCH -A es_mazza
#SBATCH -n 2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=16000
#SBATCH --gpus=rtx_3090:1



module load gcc/8.2.0 python_gpu/3.11.2
module load cuda

python run_pinn.py