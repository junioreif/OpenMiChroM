#!/bin/bash
# run run_opt_sim_step.slurm

iteration="0"
sequence="input/seq_chr10_100k.txt"
lambda="lambdas/lambda_${iteration}"
dense="input/chr10_100k_NA.dense"

sbatch run_opt_sim_step_types.slurm $sequence $lambda $iteration $dense

