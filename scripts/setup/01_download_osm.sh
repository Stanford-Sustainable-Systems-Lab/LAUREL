#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=0-01:00
#SBATCH --job-name=download_osm
#SBATCH --output=logs/slurm/%x_%j.log

# Downloads the OpenStreetMap PBF extract for North America, which both the
# compute_routes pipeline (as Graphhopper's road network) and the describe_locations
# pipeline (as a source of truck stops and warehouses) read.
#
# Everything about the transfer is configured on the osm_north_america entry in
# conf/base/catalog.yml: the source URL, and the destination, which follows
# --params=data_dir like every other dataset. Bump the snapshot date there.
#
# Re-running this is cheap: an existing file is left alone rather than re-fetched, so
# this step is safe to include when resuming the chain. To force a fresh copy, delete
# the file first. The transfer is ~18.5 GB at ~14 MB/s, so budget ~25 minutes.

# Point OpenSSL at this machine's CA bundle. The venv's interpreter is uv-managed
# (python-build-standalone) and links its own OpenSSL, whose compiled-in trust paths are
# /etc/ssl/cert.pem and c_rehash-style hashed symlinks in /etc/ssl/certs -- neither of
# which exists on Sherlock's RHEL7 image, where the bundle is a single file at
# /etc/pki/tls/cert.pem. ssl.create_default_context() therefore loads *zero* CA
# certificates, and every HTTPS request through aiohttp -- which is what fsspec's HTTP
# filesystem, and so DownloadedFileDataset, runs on -- dies with
# CERTIFICATE_VERIFY_FAILED "unable to get local issuer certificate". (requests and
# httpx default to certifi and would not have noticed; neither did the `wget` this step
# used before the download moved into the pipeline.) An SSL_CERT_FILE already in the
# environment wins, and a machine whose Python finds its own certs -- a Mac, or the
# Debian-based container -- matches nothing here and is left alone.
if [[ -z "${SSL_CERT_FILE:-}" ]]; then
  for bundle in /etc/pki/tls/cert.pem /etc/ssl/certs/ca-certificates.crt; do
    if [[ -r "$bundle" ]]; then
      export SSL_CERT_FILE="$bundle"
      break
    fi
  done
fi

uv run kedro run --pipeline=download_inputs --tags=osm --params=data_dir=$SCRATCH/laurel/data
