# Cloud Handoff Instructions (Kaggle)

If the local pilot decision rule (in `runs/pilot/PILOT_DECISION.md`) recommends proceeding, the codebase is ready to scale on Kaggle.

## Scaling Up

To run the full matrix:
- **300-500 examples**: Modify `src/dataset_generator.py` to loop 60-100 times instead of 6, possibly introducing randomized slot values instead of hardcoded strings to ensure diversity.
- **Multiple models**: The `scripts/run_pilot.py` script already accepts `--model`. You can loop over models (e.g., `meta-llama/Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.2`).
- **Multiple random seeds**: Pass `--seed` or set `RANDOM_SEED` in a loop.
- **Longer conversations**: Extend the templates in `dataset_generator.py` to include more turns.

## Kaggle Execution Commands

1. **Setup Environment**:
   ```bash
   !pip install -r requirements.txt
   ```

2. **Generate Full Dataset**:
   ```bash
   !python scripts/generate_pilot.py
   ```

3. **Run Models (example)**:
   ```bash
   !python scripts/run_pilot.py --model meta-llama/Meta-Llama-3-8B-Instruct --strategies full_history recent_turns active_state garbage_collected_history
   !python scripts/run_pilot.py --model Qwen/Qwen2.5-7B-Instruct --strategies full_history recent_turns active_state garbage_collected_history
   ```

4. **Evaluate and Analyze**:
   ```bash
   !python scripts/evaluate_pilot.py
   !python scripts/inspect_errors.py
   ```

Download `runs/pilot/metrics.json` and `runs/pilot/error_analysis.md` for final reporting.
