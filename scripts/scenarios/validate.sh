#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=0-01:00
#SBATCH --array=0-0
#SBATCH --job-name=validate
#SBATCH --output=logs/slurm/%x_%A_%a.log

cd conf && conf_dir=$(find scenarios/validate -type d -name "task_$SLURM_ARRAY_TASK_ID") && cd ..
KEDRO_CONTAINER_DATA_DIR=$SCRATCH/laurel/data ./scripts/setup/run-in-container.sh kedro run --pipeline=electrify_trips --env=$conf_dir
KEDRO_CONTAINER_DATA_DIR=$SCRATCH/laurel/data ./scripts/setup/run-in-container.sh kedro run --pipeline=evaluate_impacts --env=$conf_dir