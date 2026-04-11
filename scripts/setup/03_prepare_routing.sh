#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=0-00:10
#SBATCH --job-name=prepare_routing
#SBATCH --output=logs/slurm/%x_%j.log

uv run kedro run --pipeline=compute_routes --tags=pre_routing --params=partition_data_dir=$SCRATCH/laurel