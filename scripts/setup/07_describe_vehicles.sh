#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=0-00:10
#SBATCH --job-name=describe_vehicles
#SBATCH --output=logs/slurm/%x_%j.log

uv run kedro run --pipeline=describe_vehicles --params=data_dir=$SCRATCH/laurel/data