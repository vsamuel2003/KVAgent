#!/bin/bash
# Run tau2 airline domain benchmark with Qwen3-4B and extract profiling CSVs.
#
# Usage:
#   bash scripts/run_baseline_profiling.sh
#
# Environment variables (optional overrides):
#   MODEL      - HuggingFace model ID (default: Qwen/Qwen3-4B)
#   NUM_TASKS  - Number of tasks to run (default: 5)
#   NUM_TRIALS - Number of trials per task (default: 1)
#   SAVE_DIR   - Where to write results (default: auto-generated)

set -e

MODEL=${MODEL:-"Qwen/Qwen3-4B"}
NUM_TASKS=${NUM_TASKS:-5}
NUM_TRIALS=${NUM_TRIALS:-1}

echo "=== Tau2 Baseline Profiling Run ==="
echo "Model:      $MODEL"
echo "Domain:     airline"
echo "Num tasks:  $NUM_TASKS"
echo "Num trials: $NUM_TRIALS"
echo ""

# Run benchmark
tau2 run \
    --domain airline \
    --agent-llm "$MODEL" \
    --user-llm "$MODEL" \
    --num-tasks "$NUM_TASKS" \
    --num-trials "$NUM_TRIALS" \
    --verbose-logs \
    --log-level INFO

# Find the latest results directory for the airline domain
RESULTS_PATH=$(ls -td data/results/airline_* 2>/dev/null | head -1)/results.json

if [ ! -f "$RESULTS_PATH" ]; then
    echo "ERROR: Could not find results file at $RESULTS_PATH"
    echo "Check data/results/ for the output directory."
    exit 1
fi

echo ""
echo "Results file: $RESULTS_PATH"
echo ""

# Extract profiling CSVs
python -m tau2.scripts.extract_profiling \
    --results-path "$RESULTS_PATH" \
    --summary-output baseline_profiling_summary.csv \
    --detailed-output baseline_profiling_detailed.csv

echo ""
echo "=== Done ==="
echo "Summary CSV:  baseline_profiling_summary.csv"
echo "Detailed CSV: baseline_profiling_detailed.csv"
