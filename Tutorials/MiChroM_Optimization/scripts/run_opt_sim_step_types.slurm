#!/bin/bash -l

#SBATCH --job-name=sim_opt
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=2
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --export=ALL

echo "Submitting simulations..."

module purge

# On ARIES
module load foss/2020b Launcher_GPU OpenMPI 

# Locan conda environment
source $HOME/anaconda3/bin/activate
conda activate openmm

# On Aries, submit nodes*8GPUs simulations
max_replicas=$((SLURM_NNODES*8))

# Controling Launcher_GPU
export LAUNCHER_WORKDIR=`pwd`
export LAUNCHER_JOB_FILE=$PWD/launcher_jobs_sim
export LAUNCHER_BIND=1

# I/O control
export IN=$PWD

# and showing some job info
echo "Job started on " `date`
echo "Running on hostname" `hostname`
echo "Job $SLURM_JOB_ID is running on: $SLURM_NODELIST"
echo "Job SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR"
echo "Running on $SLURM_NNODES nodes"
echo "Running on $SLURM_NPROCS processors"
echo "CPUS per task is $SLURM_CPUS_PER_TASK"
echo "LAUNCHER_WORKDIR: $LAUNCHER_WORKDIR"
echo "IN dir is $IN"
echo "Number of replicas is $max_replicas"
df -h

# args: sequence_file iteration_number dense_file
sequence=$1
iteration=$2
dense=$3

while [ $iteration -lt 40 ] && [ -e ./lambdas/lambda_$iteration ]; 
do

	rm $IN/launcher_jobs_sim &> /dev/null

	lambda="lambda_${iteration}"

	# Create Launcher_job_file
	for i in `seq 1 $max_replicas`
	do
		mkdir output_${iteration}_${i}
		echo "python run_opt_sim_types.py $i $sequence ./lambdas/$lambda output_${iteration}_${i}/ ${iteration} > output_${iteration}_${i}/output_${iteration}_${i}.log" >> $IN/launcher_jobs_sim
	done

	if [[ ! -e output_${iteration}_1/output_${iteration}_1.log ]]; then
		$LAUNCHER_DIR/paramrun
	else
		echo "File output_${iteration}_1/output_${iteration}_1.log already exists"
	fi

	mkdir hic phi &> /dev/null

	python run_opt_step_types.py $iteration ./lambdas/ $sequence $dense output_${iteration}_*

	iteration=$(($iteration+1))
    echo "Next_iteration is $iteration"

	rm $IN/launcher_jobs_sim &> /dev/null

	echo "My job finished at:" `date`
done