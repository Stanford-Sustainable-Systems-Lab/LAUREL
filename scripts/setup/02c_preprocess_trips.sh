#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=0-00:30
#SBATCH --job-name=preprocess_trips
#SBATCH --output=logs/slurm/%x_%j.log

uv run kedro run --pipeline=describe_dwells --tags=format_trips --params=partition_data_dir=$SCRATCH/laurel