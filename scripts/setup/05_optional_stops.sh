#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=0-02:00
#SBATCH --job-name=optional_stops
#SBATCH --output=logs/slurm/%x_%j.log

uv run kedro run --pipeline=compute_routes --tags=insert_optional_stops --params=partition_data_dir=$SCRATCH/laurel