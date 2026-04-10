# logs/slurm/

This directory is tracked in git even though its contents are gitignored.

SLURM writes job output and error files here (e.g. `slurm-<job_id>_<task_id>.out`).
If the directory does not exist when a job starts, SLURM will fail to open its log
files — often silently, making the failure hard to diagnose. Tracking this directory
via `.gitkeep` ensures it is present on a fresh clone before any jobs are submitted.

The top-level `logs/` directory (used by GraphHopper) is **not** tracked.
