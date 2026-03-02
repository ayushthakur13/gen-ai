import csv
import json
from statistics import mean
from collections import Counter

from src.prompts import zero_shot, role_based, few_shot, structured_output
from src.groq_client import call_llm

# Number of times each input is repeated (for consistency testing)
RUNS_PER_INPUT = 3

STRATEGIES = {
    "zero_shot": zero_shot,
    "role_based": role_based,
    "few_shot": few_shot,
    "structured": structured_output,
}


def check_compliance(strategy_name, output):
    """
    Checks whether model output follows required format.
    """
    output = output.strip()

    if strategy_name == "structured":
        try:
            parsed = json.loads(output)
            return (
                isinstance(parsed, dict)
                and "sentiment" in parsed
                and parsed["sentiment"] in ["Positive", "Negative"]
            )
        except:
            return False
    else:
        return output in ["Positive", "Negative"]


def run_experiments(test_inputs):
    summary_stats = {}

    with open("logs/prompt_logs.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "strategy",
            "input",
            "run_number",
            "output",
            "prompt_tokens",
            "completion_tokens",
            "latency",
            "format_compliant"
        ])

        for strategy_name, strategy_func in STRATEGIES.items():

            all_latencies = []
            all_prompt_tokens = []
            all_completion_tokens = []
            compliance_flags = []
            consistency_scores = []

            print(f"\nRunning strategy: {strategy_name}")

            for text in test_inputs:

                outputs = []

                for run in range(RUNS_PER_INPUT):

                    prompt = strategy_func(text)
                    result = call_llm(prompt)

                    compliant = check_compliance(strategy_name, result["output"])

                    writer.writerow([
                        strategy_name,
                        text,
                        run + 1,
                        result["output"],
                        result["prompt_tokens"],
                        result["completion_tokens"],
                        result["latency"],
                        compliant
                    ])

                    outputs.append(result["output"].strip())

                    all_latencies.append(result["latency"])
                    all_prompt_tokens.append(result["prompt_tokens"])
                    all_completion_tokens.append(result["completion_tokens"])
                    compliance_flags.append(compliant)

                # Consistency score for this input
                most_common_count = Counter(outputs).most_common(1)[0][1]
                consistency = most_common_count / RUNS_PER_INPUT
                consistency_scores.append(consistency)

            # Strategy-level summary
            summary_stats[strategy_name] = {
                "avg_latency": mean(all_latencies),
                "avg_prompt_tokens": mean(all_prompt_tokens),
                "avg_completion_tokens": mean(all_completion_tokens),
                "compliance_rate": sum(compliance_flags) / len(compliance_flags),
                "avg_consistency": mean(consistency_scores),
            }

    print("\n===== SUMMARY =====")
    for strategy, stats in summary_stats.items():
        print(f"\nStrategy: {strategy}")
        print(f"Average Latency: {stats['avg_latency']:.3f}s")
        print(f"Average Prompt Tokens: {stats['avg_prompt_tokens']:.2f}")
        print(f"Average Completion Tokens: {stats['avg_completion_tokens']:.2f}")
        print(f"Format Compliance Rate: {stats['compliance_rate']*100:.2f}%")
        print(f"Average Consistency: {stats['avg_consistency']*100:.2f}%")
