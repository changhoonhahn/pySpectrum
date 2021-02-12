#!/bin/bash
#SBATCH -J bk_glam
#SBATCH --output=_bk_glam.o
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=40
#SBATCH --partition=general
#SBATCH --time=10:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=all
#SBATCH --mail-user=chhahn@princeton.edu

module load anaconda3
source activate emanu

for i in {1..15}; do 
    srun -n 1 --mpi=pmi2 python3 glam.py $i 13 False
    #srun -n 1 --mpi=pmi2 python3 glam.py 1 13 True 
done 

