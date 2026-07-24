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
    errors = []
    
    for raw_file in raw_files:
        with open(raw_file, "r", encoding="utf-8") as f:
            output = ModelOutput.model_validate_json(f.read())
            
        ex_id = "_".join(os.path.basename(raw_file).split("_")[:3])
        if ex_id not in examples:
            continue
            
        ex = examples[ex_id]
        res = evaluator.evaluate_example(ex, output)
        
        is_error = False
        reasons = []
        
        if res["tool_accuracy"] == 0:
            is_error = True
            reasons.append("tool_accuracy_fail")
        if res["arg_exact_match"] == 0:
            is_error = True
            reasons.append("arg_exact_match_fail")
        if res["stale_reuse"] == 1:
            is_error = True
            reasons.append("stale_reuse")
        if res["accidental_deletion"] == 1:
            is_error = True
            reasons.append("accidental_deletion")
        if res["cancellation_failure"] == 1:
            is_error = True
            reasons.append("cancellation_failure")
            
        if is_error:
            errors.append({
                "example": ex,
                "output": output,
                "reasons": reasons
            })
            
    with open("runs/pilot/error_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Error Analysis\n\n")
        f.write(f"Total Errors Found: {len(errors)}\n\n")
        
        for err in errors:
            ex = err["example"]
            out = err["output"]
            f.write(f"## Example: {ex.example_id} | Strategy: {out.strategy}\n")
            f.write(f"**Failure Reasons:** {', '.join(err['reasons'])}\n\n")
            
            f.write("### Conversation\n")
            for t in ex.conversation_turns:
                f.write(f"- **{t.role}**: {t.content}\n")
                
            f.write("\n### Final Gold State\n")
            f.write("```json\n")
            f.write(json.dumps(ex.final_active_state, indent=2))
            f.write("\n```\n")
            
            f.write("### Superseded Values\n")
            f.write("```json\n")
            f.write(json.dumps(ex.superseded_values, indent=2))
            f.write("\n```\n")
            
            f.write("### Model Parsed Output\n")
            f.write("```json\n")
            parsed = {"tool": out.tool, "arguments": out.arguments, "clarification": out.clarification}
            f.write(json.dumps(parsed, indent=2))
            f.write("\n```\n")
            
            f.write("### Raw Output\n")
            f.write("```\n")
            f.write(out.raw_output)
            f.write("\n```\n")
            f.write("---\n\n")
            
    print(f"Error inspection completed. {len(errors)} errors found. Saved to runs/pilot/error_analysis.md")

if __name__ == "__main__":
    main()
