#!/bin/bash -l

#SBATCH --job-name=opt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH --gres=gpu:1 
#SBATCH --time=10:00:00

i=${SLURM_ARRAY_TASK_ID}


# runSim needs args --> $1: replica number $2: sequence file $3:lambda file $4: outputfolder name 
python runSim.py $i ../input/seq_chr10_100k.txt ../input/lambda_0 output_"$i"


#To run using arrays uses the following command
#1-20:1 means submit 20 jobs where the variable $i ranges from 1 to 20 

#sbatch --array=1-20:1 runIteraction.slurm 

