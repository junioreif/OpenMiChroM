#!/bin/bash -l

#SBATCH --job-name=opt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH --gres=gpu:1 
#SBATCH --time=10:00:00


# runOpt needs args --> $1: iteraction step  $2: input folder $3:sequence file
# $4: dense file $5 all output folder

# These cluster scripts use the full 1,356-bin sequence, not the notebook sample.
full_map=../../optimization_data/chr10_100k.dense
if [[ ! -f "$full_map" ]]; then
    echo "Download the original chr10_100k.dense as described in Tutorial_MiChroM_Optimization.ipynb." >&2
    exit 1
fi
python runOpt.py 1 ../input/ ../input/seq_chr10_100k.txt "$full_map" output_*

