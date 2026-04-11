#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --time=1-00:00
#SBATCH --job-name=download_osm
#SBATCH --output=logs/slurm/%x_%j.log

wget -P $SCRATCH/north-america-latest.osm.pbf https://download.geofabrik.de/north-america-latest.osm.pbf