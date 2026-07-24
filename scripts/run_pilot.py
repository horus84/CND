import sys
import os
import argparse
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset_generator import generate_examples
from src.context_strategies import STRATEGIES
from src.model_runner import ModelRunner
from src.schemas import PilotExample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--strategies", nargs="+", default=["full_history", "recent_turns", "active_state", "garbage_collected_history"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    out_dir = "runs/pilot/raw"
    os.makedirs(out_dir, exist_ok=True)
    
    examples_path = "data/pilot/pilot_examples.jsonl"
    if not os.path.exists(examples_path):
        print("Generating examples...")
        examples = generate_examples()
    else:
        examples = []
        with open(examples_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(PilotExample.model_validate_json(line))
                
    tasks = []
    for ex in examples:
        for strat in args.strategies:
            out_file = os.path.join(out_dir, f"{ex.example_id}_{strat}.json")
            if args.resume and os.path.exists(out_file):
                continue
            tasks.append((ex, strat, out_file))
            
    if not tasks:
        print("All tasks completed.")
        return
        
    print(f"Running {len(tasks)} tasks...")
    runner = ModelRunner(model_name=args.model)
    
    for ex, strat, out_file in tqdm(tasks):
        strat_fn = STRATEGIES[strat]
        prompt, _ = strat_fn(ex)
        
        output = runner.generate(prompt, strat)
        
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
