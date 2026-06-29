#!/bin/bash
# =============================================================================
# Docker entrypoint for Aerial Lab container
# Ensures the aeriallab conda environment is active before running any command.
# =============================================================================

set -e

# Source conda
source /opt/conda/etc/profile.d/conda.sh
conda activate aeriallab

# If the first argument looks like a flag or a script, run it with bash
# Otherwise exec the command directly (e.g. "bash", "python scripts/...")
exec "$@"
