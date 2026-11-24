#!/bin/zsh
# End-to-end reproduction: every stage, every table and figure (PNG + PDF). Resumable; finished
# outputs are reused. Options pass through, e.g.  ./run_all.sh --stages figures   or   --dry-run
# See seams/experiments/pipeline.py for the stages.
cd "${0:A:h}"
PY="${SEAMS_PY:-python}"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
exec "$PY" -m seams.experiments.pipeline "$@"
