#!/usr/bin/env bash
# Run the "Good Life" deliberation with all six educated traditions.
#
# Usage:
#   ./run_good_life.sh                    # Default: Claude, 5 rounds, with knowledge
#   ./run_good_life.sh --mock             # Test run without API key
#   ./run_good_life.sh --rounds 3         # Fewer rounds
#   ./run_good_life.sh --quiet            # Suppress live output
#
# Outputs saved to:
#   good_life_report.txt   — Human-readable report
#   good_life_data.json    — Raw JSON for analysis

cd "$(dirname "$0")"

python3 polis.py \
    --scenario scenarios/good_life.json \
    --knowledge \
    --rounds 3 \
    --output good_life_report.txt \
    --json good_life_data.json \
    "$@"
