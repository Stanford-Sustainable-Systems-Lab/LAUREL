#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=0-00:05
#SBATCH --job-name=prepare_totals
#SBATCH --output=logs/slurm/%x_%j.log

uv run kedro run --pipeline=prepare_totals --params=partition_data_dir=$SCRATCH/laurel