#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=0-01:00
#SBATCH --job-name=describe_locations
#SBATCH --output=logs/slurm/%x_%j.log

uv run kedro run --pipeline=describe_locations --params=partition_data_dir=$SCRATCH/laurel