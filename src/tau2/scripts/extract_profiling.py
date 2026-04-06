"""
Extract profiling data from tau2 results JSON and output to CSV.

Usage:
    python -m tau2.scripts.extract_profiling \
        --results-path path/to/results.json \
        --summary-output profiling_summary.csv \
        --detailed-output profiling_detailed.csv
"""
import argparse
import csv
import json
from pathlib import Path


def load_results(results_path: Path) -> dict:
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def extract_summary(simulations: list[dict]) -> list[dict]:
    """One row per simulation."""
    rows = []
    for sim in simulations:
        profiling = (sim.get("info") or {}).get("profiling") or {}
        aggregates = profiling.get("aggregates", {})
        reward_info = sim.get("reward_info") or {}
        reward = reward_info.get("reward")

        rows.append({
            "task_id": sim.get("task_id", ""),
            "trial": sim.get("trial", 0),
            "duration_seconds": sim.get("duration", 0.0),
            "num_steps": profiling.get("num_steps", 0),
            "avg_step_latency_seconds": aggregates.get("avg_step_latency_seconds", 0.0),
            "total_llm_time_seconds": aggregates.get("total_llm_time_seconds", 0.0),
            "total_tool_stall_time_seconds": aggregates.get("total_tool_stall_time_seconds", 0.0),
            "avg_llm_latency_seconds": aggregates.get("avg_llm_latency_seconds", 0.0),
            "total_prompt_tokens": aggregates.get("total_prompt_tokens", 0),
            "total_completion_tokens": aggregates.get("total_completion_tokens", 0),
            "reward": reward,
            "termination_reason": sim.get("termination_reason", ""),
        })
    return rows


def extract_detailed(simulations: list[dict]) -> list[dict]:
    """One row per step per simulation."""
    rows = []
    for sim in simulations:
        profiling = (sim.get("info") or {}).get("profiling") or {}
        task_id = sim.get("task_id", "")
        trial = sim.get("trial", 0)

        for step in profiling.get("steps", []):
            tool_execs = step.get("tool_executions", [])
            tool_stall = sum(
                e.get("execution_time_seconds") or 0.0
                for e in tool_execs
                if e.get("execution_time_seconds") is not None
            )
            rows.append({
                "task_id": task_id,
                "trial": trial,
                "step_idx": step.get("step_idx", 0),
                "wall_time_seconds": step.get("wall_time_seconds"),
                "llm_latency_seconds": step.get("llm_latency_seconds"),
                "prompt_tokens": step.get("prompt_tokens"),
                "completion_tokens": step.get("completion_tokens"),
                "num_tool_calls": len(tool_execs),
                "tool_stall_time_seconds": tool_stall,
            })
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        print(f"No data to write to {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract profiling data from tau2 results.")
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to the tau2 results.json file",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("profiling_summary.csv"),
        help="Output path for the summary CSV (one row per simulation)",
    )
    parser.add_argument(
        "--detailed-output",
        type=Path,
        default=Path("profiling_detailed.csv"),
        help="Output path for the detailed CSV (one row per step)",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.results_path} ...")
    data = load_results(args.results_path)
    simulations = data.get("simulations", [])
    print(f"Found {len(simulations)} simulations")

    summary_rows = extract_summary(simulations)
    detailed_rows = extract_detailed(simulations)

    write_csv(summary_rows, args.summary_output)
    write_csv(detailed_rows, args.detailed_output)

    # Print a quick summary to stdout
    if summary_rows:
        total_steps = sum(r["num_steps"] for r in summary_rows)
        avg_duration = sum(r["duration_seconds"] for r in summary_rows) / len(summary_rows)
        avg_llm = sum(r["total_llm_time_seconds"] for r in summary_rows) / len(summary_rows)
        avg_tool = sum(r["total_tool_stall_time_seconds"] for r in summary_rows) / len(summary_rows)
        print(f"\n=== Profiling Summary ===")
        print(f"Simulations: {len(summary_rows)}")
        print(f"Avg episode duration: {avg_duration:.2f}s")
        print(f"Total steps across all episodes: {total_steps}")
        print(f"Avg total LLM time per episode: {avg_llm:.2f}s")
        print(f"Avg total tool stall per episode: {avg_tool:.2f}s")


if __name__ == "__main__":
    main()
