import json
from typing import Tuple, List, Dict, Any
from .schemas import PilotExample, ConversationTurn

def apply_full_history(example: PilotExample) -> Tuple[str, int]:
    turns = []
    for t in example.conversation_turns:
        turns.append(f"{t.role.capitalize()}: {t.content}")
    prompt = "\n".join(turns)
    return prompt, len(prompt.split())

def apply_recent_turns(example: PilotExample) -> Tuple[str, int]:
    turns = []
    recent = example.conversation_turns[-4:]
    for t in recent:
        turns.append(f"{t.role.capitalize()}: {t.content}")
    prompt = "\n".join(turns)
    return prompt, len(prompt.split())

def apply_active_state(example: PilotExample) -> Tuple[str, int]:
    lines = ["--- ACTIVE STATE ---"]
    lines.append(json.dumps(example.final_active_state, indent=2))
    
    if example.ambiguity_status:
        lines.append("--- UNRESOLVED AMBIGUITIES ---")
        lines.append("Ambiguity present.")
        
    lines.append("--- CURRENT USER REQUEST ---")
    # Include ONLY the last user turn, avoiding assistant turns that may leak superseded values
    last_user_turn = [t for t in example.conversation_turns if t.role == "user"][-1]
    lines.append(f"User: {last_user_turn.content}")
        
    prompt = "\n".join(lines)
    return prompt, len(prompt.split())

def apply_garbage_collected_history(example: PilotExample) -> Tuple[str, int]:
    lines = ["--- REVISED HISTORY ---"]
    
    if example.final_active_state:
        lines.append("Current active facts:")
        for k, v in example.final_active_state.items():
            lines.append(f"- {k}: {v}")
    
    if example.superseded_values:
        lines.append("\nSuperseded (OUTDATED) facts:")
        for k, v_list in example.superseded_values.items():
            for v in v_list:
                lines.append(f"- {k}: {v} [SUPERSEDED]")
                
    lines.append("\n--- LATEST INSTRUCTIONS ---")
    recent = example.conversation_turns[-2:]
    for t in recent:
        lines.append(f"{t.role.capitalize()}: {t.content}")
        
    prompt = "\n".join(lines)
    return prompt, len(prompt.split())

def apply_oracle_clean_history(example: PilotExample) -> Tuple[str, int]:
    turns = []
    for t in example.oracle_clean_turns:
        turns.append(f"{t.role.capitalize()}: {t.content}")
    prompt = "\n".join(turns)
    return prompt, len(prompt.split())

STRATEGIES = {
    "full_history": apply_full_history,
    "recent_turns": apply_recent_turns,
    "active_state": apply_active_state,
    "garbage_collected_history": apply_garbage_collected_history,
    "oracle_clean_history": apply_oracle_clean_history
}
