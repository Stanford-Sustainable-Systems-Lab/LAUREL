#!/bin/bash
# Submits the scripts/setup/*.sh pipeline as a chain of SLURM jobs, each
# depending on the previous one via --dependency=afterok. If any step fails
# or is cancelled, SLURM automatically cancels the remaining downstream jobs.
#
# Usage:
#   ./submit_all.sh                          # submit the full 01->08 chain
#   ./submit_all.sh --from=04_compute_routes.sh   # resume from a later step
set -euo pipefail
cd "$(dirname "$0")"

STEPS=(
  01_download_osm.sh
  02a_describe_locations.sh
  02b_import_graph.sh
  02c_preprocess_trips.sh
  03_prepare_routing.sh
  04_compute_routes.sh
  05_optional_stops.sh
  06_describe_dwells.sh
  07_describe_vehicles.sh
  08_prepare_totals.sh
)

start_idx=0
if [[ "${1:-}" == --from=* ]]; then
  from="${1#--from=}"
  start_idx=-1
  for i in "${!STEPS[@]}"; do
    if [[ "${STEPS[$i]}" == "$from" ]]; then
      start_idx=$i
      break
    fi
  done
  if [[ $start_idx -eq -1 ]]; then
    echo "Unknown step '$from'. Valid steps are:" >&2
    printf '  %s\n' "${STEPS[@]}" >&2
    exit 1
  fi
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--from=<step_script_name>]" >&2
  exit 1
fi

prev_jid=""
for step in "${STEPS[@]:$start_idx}"; do
  if [[ -z "$prev_jid" ]]; then
    jid=$(sbatch --parsable "$step")
  else
    jid=$(sbatch --parsable --dependency=afterok:"$prev_jid" "$step")
  fi
  echo "$step -> $jid"
  prev_jid=$jid
done
