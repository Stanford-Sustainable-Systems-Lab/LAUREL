#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=0-03:00
#SBATCH --job-name=describe_locations
#SBATCH --output=logs/slurm/%x_%j.log

# Runs the full describe_locations pipeline, including the one-time NLCD raster
# extraction (read_land_use). This step is I/O-intensive and may take several
# hours; resources are sized accordingly.
#
# To re-run only the freight-activity-class clustering without repeating the 
# raster extraction or other heavy preprocessing, add the following argument:
#   --tags=fast_loc_grouping

uv run kedro run --pipeline=describe_locations --params=partition_data_dir=$SCRATCH/laurel
