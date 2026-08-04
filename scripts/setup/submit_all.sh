#!/bin/bash
# Submits the scripts/setup/*.sh pipeline as a chain of SLURM jobs, each
# depending on the previous one via --dependency=afterok. If any step fails
# or is cancelled, SLURM automatically cancels the remaining downstream jobs.
#
# Usage (from the repository root):
#   ./scripts/setup/submit_all.sh                        # submit the full 01->08 chain
#   ./scripts/setup/submit_all.sh --from=04_compute_routes.sh  # resume from a later step
#   ./scripts/setup/submit_all.sh --dry-run              # print the sbatch commands only
#   ./scripts/setup/submit_all.sh --mail-user=you@stanford.edu  # email on each step's end/failure
#   ./scripts/setup/submit_all.sh --mail-user=you@stanford.edu --mail-type=ALL  # add BEGIN too

# Must run as a subprocess: this script uses `set -e`, `exit`, and positional
# args, none of which behave correctly when sourced into an interactive shell.
# This check has to precede `set -euo pipefail`, or sourcing arms errexit in
# the caller's shell before we get a chance to bail out.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run this script, don't source it:  ${BASH_SOURCE[0]}" >&2
  return 1
fi

set -euo pipefail

# Submit from the repo root, not from this script's directory. The step scripts
# use a relative `#SBATCH --output=logs/slurm/%x_%j.log`, which SLURM resolves
# against the job's working directory (inherited from wherever sbatch ran), so
# submitting from scripts/setup/ would scatter logs into scripts/setup/logs/.
STEPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STEPS_DIR/../.."
mkdir -p logs/slurm

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

from=""
dry_run=0
mail_user=""
mail_type="END,FAIL"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from=*) from="${1#--from=}" ;;
    --dry-run) dry_run=1 ;;
    --mail-user=*) mail_user="${1#--mail-user=}" ;;
    --mail-type=*) mail_type="${1#--mail-type=}" ;;
    *)
      echo "Usage: $0 [--from=<step_script_name>] [--dry-run]" \
           "[--mail-user=<address>] [--mail-type=<BEGIN,END,FAIL,ALL,...>]" >&2
      exit 1
      ;;
  esac
  shift
done

# Catch a mistyped address now rather than after ten jobs have been queued with
# notifications silently going nowhere.
if [[ -n "$mail_user" && "$mail_user" != *@* ]]; then
  echo "--mail-user='$mail_user' does not look like an email address." >&2
  exit 1
fi
if [[ -z "$mail_user" && "$mail_type" != "END,FAIL" ]]; then
  echo "--mail-type has no effect without --mail-user." >&2
  exit 1
fi


start_idx=0
if [[ -n "$from" ]]; then
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
fi

prev_jid=""
for step in "${STEPS[@]:$start_idx}"; do
  args=(--parsable)
  if [[ -n "$prev_jid" ]]; then
    args+=(--dependency=afterok:"$prev_jid")
  fi
  # sbatch command-line flags win over the step scripts' #SBATCH directives, so
  # notifications stay opt-in per submission instead of being baked into a file.
  if [[ -n "$mail_user" ]]; then
    args+=(--mail-user="$mail_user" --mail-type="$mail_type")
  fi
  args+=("$STEPS_DIR/$step")

  if (( dry_run )); then
    echo "sbatch ${args[*]}"
    # Stand-in so the next step's --dependency reads sensibly.
    prev_jid="<jobid:$step>"
  else
    jid=$(sbatch "${args[@]}")
    echo "$step -> $jid"
    prev_jid=$jid
  fi
done
