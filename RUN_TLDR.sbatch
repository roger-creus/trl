#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 4
#SBATCH --time=46:00:00
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --account=m4881
#SBATCH --output=/pscratch/sd/r/$(whoami)/%x-%j.out

module load conda
conda activate trl

cd /global/homes/r/rogercc/trl
sh run_all_tldr.sh

scontrol update jobid=$SLURM_JOB_ID state=cancelled