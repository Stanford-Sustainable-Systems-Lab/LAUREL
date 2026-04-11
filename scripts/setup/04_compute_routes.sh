#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=0-10:00
#SBATCH --job-name=compute_routes
#SBATCH --output=logs/slurm/%x_%j.log

uv run kedro run --pipeline=compute_routes --tags=routing --params=partition_data_dir=$SCRATCH/laurel