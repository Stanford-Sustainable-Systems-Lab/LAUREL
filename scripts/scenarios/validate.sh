#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=0-01:15
#SBATCH --array=0-0
#SBATCH --job-name=validate
#SBATCH --output=logs/slurm/%x_%A_%a.log

cd conf && conf_dir=$(find scenarios/validate -type d -name "task_$SLURM_ARRAY_TASK_ID") && cd ..
uv run kedro run --pipeline=electrify_trips --env=$conf_dir --params=partition_data_dir=$SCRATCH/laurel
uv run kedro run --pipeline=evaluate_impacts --env=$conf_dir --params=partition_data_dir=$SCRATCH/laurel