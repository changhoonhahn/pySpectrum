#!/bin/bash
#SBATCH -p debug 
#SBATCH -n 1
#SBATCH -t 00:15:00 
#SBATCH -J test_pyspec 
#SBATCH -o _test_pyspec.o

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0
module load cray-fftw/3.3.8.1

export OMP_NUM_THREADS=1
#srun -n 1 python /global/homes/c/chahah/projects/pySpectrum/run/test_pyspec.py
python /global/homes/c/chahah/projects/pySpectrum/run/test_pyspec.py

now=$(date +"%T") 
echo "end time ... $now"
