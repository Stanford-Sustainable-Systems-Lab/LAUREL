#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=160GB
#SBATCH --time=0-04:00
#SBATCH --job-name=import_graph
#SBATCH --output=logs/slurm/%x_%j.log

uv run kedro run --pipeline=compute_routes --tags=import --params=partition_data_dir=$SCRATCH/laurel