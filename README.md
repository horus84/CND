# Context Has No Delete Key

A research prototype investigating whether LLM agents reuse superseded values when users correct, overwrite, cancel, delete, or roll back information during a multi-turn interaction.

## Project Structure

- `src/`: Core logic (state tracking, context strategies, dataset generation, model running, evaluation).
- `scripts/`: CLI entrypoints.
- `data/pilot/`: Generated pilot datasets.
- `runs/pilot/`: Checkpoints and evaluation results.
- `tests/`: Unit tests for critical components.

## Running the Pilot Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the 30-example deterministic pilot dataset:
   ```bash
   python scripts/generate_pilot.py
   ```

3. Run the model on the dataset (supports checkpointing/resume):
   ```bash
   python scripts/run_pilot.py \
     --model Qwen/Qwen2.5-1.5B-Instruct \
     --strategies full_history recent_turns active_state garbage_collected_history \
     --resume
   ```

4. Evaluate the outputs:
   ```bash
   python scripts/evaluate_pilot.py
   ```

5. Inspect failure modes:
   ```bash
   python scripts/inspect_errors.py
   ```

After evaluation, review `runs/pilot/PILOT_DECISION.md` for go/no-go recommendations for the cloud benchmark.
