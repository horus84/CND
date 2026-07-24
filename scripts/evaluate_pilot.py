import sys
import os
import json
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.schemas import PilotExample, ModelOutput
from src.evaluator import Evaluator

def main():
    examples_path = "data/pilot/pilot_examples.jsonl"
    examples = {}
    with open(examples_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = PilotExample.model_validate_json(line)
            examples[ex.example_id] = ex
            
    raw_files = glob.glob("runs/pilot/raw/*.json")
    results = []
    
    evaluator = Evaluator()
    for raw_file in raw_files:
        with open(raw_file, "r", encoding="utf-8") as f:
            output = ModelOutput.model_validate_json(f.read())
            
        ex_id = "_".join(os.path.basename(raw_file).split("_")[:3])
        if ex_id not in examples:
            continue
            
        ex = examples[ex_id]
        res = evaluator.evaluate_example(ex, output)
        results.append(res)
        
    agg = evaluator.aggregate(results)
    
    with open("runs/pilot/metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
        
    print("Evaluation completed. Saved to runs/pilot/metrics.json")
    print("\n--- Summary of Key Metrics ---")
    
    strategies = ["full_history", "recent_turns", "garbage_collected_history", "active_state"]
    print(f"{'Strategy':<30} | {'Strict Accuracy':<15} | {'Tolerant Accuracy':<20} | {'Stale Reuse'}")
    print("-" * 85)
    for s in strategies:
        strict = agg.get(f"{s}_tool_accuracy", 0.0)
        tolerant = agg.get(f"{s}_tolerant_tool_accuracy", 0.0)
        stale = agg.get(f"{s}_stale_reuse", 0.0)
        print(f"{s:<30} | {strict*100:>14.1f}% | {tolerant*100:>19.1f}% | {stale*100:>10.1f}%")
    
    print("\n")
    
    with open("runs/pilot/PILOT_DECISION.md", "w", encoding="utf-8") as f:
        f.write("# Pilot Decision\n\n")
        f.write("Review runs/pilot/metrics.json for full results.\n\n")
        f.write("## Recommendation\n")
        
        stale_reuse_fh = agg.get("full_history_stale_reuse", 0)
        stale_reuse_as = agg.get("active_state_stale_reuse", 0)
        stale_reuse_gc = agg.get("garbage_collected_history_stale_reuse", 0)
        
        reasons = []
        if stale_reuse_fh > 0.05:
            reasons.append("Full history produces multiple genuine stale-value errors.")
        if stale_reuse_as < stale_reuse_fh or stale_reuse_gc < stale_reuse_fh:
            reasons.append("Active-state or garbage-collected context fixes several errors.")
            
        if reasons:
            f.write("Proceed to Cloud Benchmark. Reasons:\n")
            for r in reasons:
                f.write(f"- {r}\n")
        else:
            f.write("Abandon/Redesign project. Reasons:\n")
            if stale_reuse_fh == 0:
                f.write("- No stale-value errors appear in full history.\n")
            if stale_reuse_fh > 0 and stale_reuse_fh == stale_reuse_gc == stale_reuse_as:
                f.write("- All strategies behave identically.\n")

if __name__ == "__main__":
    main()
