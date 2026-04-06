#!/bin/bash
# Run tau2 airline domain benchmark with HuggingFace models and extract profiling CSVs.
#
# Usage:
#   bash scripts/run_baseline_profiling.sh
#
# Environment variables:
#   AGENT_MODEL  - Agent LLM (default: Qwen/Qwen3-4B-Instruct-2507)
#   USER_MODEL   - User simulator LLM (default: Qwen/Qwen3-4B-Instruct-2507)
#   NUM_TASKS    - Number of tasks to run (default: 5)
#   NUM_TRIALS   - Number of trials per task (default: 1)
#
# GPU setup:
#   1 GPU:  AGENT_MODEL and USER_MODEL share the GPU (serialized inference).
#   2 GPUs: The first model loaded (agent) is assigned to cuda:0, the second
#           (user) to cuda:1, automatically. Inference runs in parallel.
#             AGENT_MODEL=Qwen/Qwen3-4B-Instruct-2507 USER_MODEL=Qwen/Qwen3-4B-Instruct-2507

set -e

AGENT_MODEL=${AGENT_MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}
USER_MODEL=${USER_MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}
NUM_TASKS=${NUM_TASKS:-5}
NUM_TRIALS=${NUM_TRIALS:-1}

echo "=== Tau2 Baseline Profiling Run ==="
echo "Agent model: $AGENT_MODEL"
echo "User model:  $USER_MODEL"
echo "Domain:      airline"
echo "Num tasks:   $NUM_TASKS"
echo "Num trials:  $NUM_TRIALS"
echo ""

# Run benchmark
# --max-concurrency 1: serialize tasks (correct for shared-GPU setups).
#   With 2 GPUs, inference within a task still runs on separate devices.
tau2 run \
    --domain airline \
    --agent-llm "$AGENT_MODEL" \
    --user-llm "$USER_MODEL" \
    --num-tasks "$NUM_TASKS" \
    --num-trials "$NUM_TRIALS" \
    --max-concurrency 1 \
    --verbose-logs \
    --log-level INFO

# Find the latest results directory for the airline domain
RESULTS_PATH=$(ls -td data/simulations/20*airline* 2>/dev/null | head -1)/results.json

if [ ! -f "$RESULTS_PATH" ]; then
    echo "ERROR: Could not find results file. Check data/simulations/ for the output directory."
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
