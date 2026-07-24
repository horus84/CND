import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset_generator import generate_examples, generate_report

if __name__ == "__main__":
    out_dir = "data/pilot"
    os.makedirs(out_dir, exist_ok=True)
    
    examples = generate_examples()
    
    jsonl_path = os.path.join(out_dir, "pilot_examples.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
            
    report_path = os.path.join(out_dir, "PILOT_DATASET_REPORT.md")
    generate_report(examples, report_path)
    
    print("--------------------------------------------------")
    print(f"Generated {len(examples)} examples.")
    print(f"Dataset saved to: {jsonl_path}")
    print(f"Report saved to: {report_path}")
    print("--------------------------------------------------")
    
    with open(report_path, "r", encoding="utf-8") as f:
        print(f.read())
