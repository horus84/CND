import json
import random
import os
from collections import defaultdict
from typing import List

from src.schemas import (
    PilotExample, Domain, OperationType, ConversationTurn, StateUpdate
)
from src.state_tracker import StateTracker

RANDOM_SEED = 42

def generate_examples() -> List[PilotExample]:
    random.seed(RANDOM_SEED)
    examples = []
    
    # We will generate 6 examples for each of the 5 operation types.
    # We'll cycle through domains to ensure a mix.
    domains = list(Domain)
    
    # 1. OVERWRITE
    for i in range(6):
        domain = domains[i % len(domains)]
        tracker = StateTracker()
        
        # Turn 1
        tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="time", new_value="3 PM"))
        tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="day", new_value="Friday"))
        
        # Turn 2
        tracker.apply_update(StateUpdate(operation_type=OperationType.OVERWRITE, field="day", new_value="Saturday"))
        
        ex = PilotExample(
            example_id=f"{domain.value}_overwrite_{i+1:03d}",
            domain=domain,
            operation_type=OperationType.OVERWRITE,
            conversation_turns=[
                ConversationTurn(role="user", content="Schedule a meeting for Friday at 3 PM."),
                ConversationTurn(role="assistant", content="Acknowledged. A meeting for Friday at 3 PM."),
                ConversationTurn(role="user", content="Actually move it to Saturday.")
            ],
            initial_state={"time": "3 PM", "day": "Friday"},
            ordered_state_updates=[
                StateUpdate(operation_type=OperationType.OVERWRITE, field="day", old_value="Friday", new_value="Saturday")
            ],
            final_active_state=tracker.get_active_state(),
            superseded_values=tracker.get_superseded_values(),
            expected_tool_name=f"create_{domain.value}_event",
            expected_normalized_tool_arguments={"time": "3 PM", "day": "Saturday"},
            ambiguity_status=False,
            expected_clarification_behavior=None,
            oracle_clean_turns=[
                ConversationTurn(role="user", content="Schedule a meeting for Saturday at 3 PM.")
            ]
        )
        examples.append(ex)
        
    # 2. DELETION
    for i in range(6):
        domain = domains[(i + 1) % len(domains)]
        tracker = StateTracker()
        
        tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="target", new_value="Rahul"))
        tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="topic", new_value="Project update"))
        
        tracker.apply_update(StateUpdate(operation_type=OperationType.DELETION, field="target"))
        tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="target", new_value="Priya"))
        
        ex = PilotExample(
            example_id=f"{domain.value}_deletion_{i+1:03d}",
            domain=domain,
            operation_type=OperationType.DELETION,
            conversation_turns=[
                ConversationTurn(role="user", content="Send a message to Rahul about the project update."),
                ConversationTurn(role="assistant", content="Drafting message to Rahul."),
                ConversationTurn(role="user", content="Wait, remove Rahul and send it to Priya instead.")
            ],
            initial_state={"target": "Rahul", "topic": "Project update"},
            ordered_state_updates=[
                StateUpdate(operation_type=OperationType.DELETION, field="target", old_value="Rahul"),
                StateUpdate(operation_type=OperationType.SET, field="target", new_value="Priya")
            ],
            final_active_state=tracker.get_active_state(),
            superseded_values={"target": ["Rahul"]}, # Technically it's in deleted, but for evaluation stale value is Rahul
            expected_tool_name=f"send_{domain.value}",
            expected_normalized_tool_arguments={"target": "Priya", "topic": "Project update"},
            ambiguity_status=False,
            expected_clarification_behavior=None,
            oracle_clean_turns=[
                ConversationTurn(role="user", content="Send a message to Priya about the project update.")
            ]
        )
        # Fix superseded to include deleted items for evaluating stale reuse
        superseded = tracker.get_superseded_values()
        for k, tomb in tracker.deleted.items():
            if k not in superseded:
                superseded[k] = []
            superseded[k].append(tomb.previous_value)
        ex.superseded_values = superseded
        
        examples.append(ex)

    # 3. CANCELLATION
    for i in range(6):
        domain = domains[(i + 2) % len(domains)]
        tracker = StateTracker()
        
        tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="item", new_value="Laptop"))
        tracker.apply_update(StateUpdate(operation_type=OperationType.CANCELLATION, new_value="purchase"))
        
        ex = PilotExample(
            example_id=f"{domain.value}_cancellation_{i+1:03d}",
            domain=domain,
            operation_type=OperationType.CANCELLATION,
            conversation_turns=[
                ConversationTurn(role="user", content="I want to buy a Laptop."),
                ConversationTurn(role="assistant", content="Proceeding to checkout for Laptop."),
                ConversationTurn(role="user", content="Cancel that, I changed my mind.")
            ],
            initial_state={"item": "Laptop"},
            ordered_state_updates=[
                StateUpdate(operation_type=OperationType.CANCELLATION, new_value="purchase")
            ],
            final_active_state=tracker.get_active_state(),
            superseded_values={"item": ["Laptop"]}, # Stale because it shouldn't be executed
            expected_tool_name=None,
            expected_normalized_tool_arguments={},
            ambiguity_status=False,
            expected_clarification_behavior="acknowledge_cancellation",
            oracle_clean_turns=[
                ConversationTurn(role="user", content="Nevermind, I don't want to buy anything.")
            ]
        )
        examples.append(ex)

    # 4. ROLLBACK
    for i in range(6):
        domain = domains[(i + 3) % len(domains)]
        tracker = StateTracker()
        
        tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="destination", new_value="Paris"))
        tracker.apply_update(StateUpdate(operation_type=OperationType.OVERWRITE, field="destination", new_value="London"))
        tracker.apply_update(StateUpdate(operation_type=OperationType.ROLLBACK))
        
        ex = PilotExample(
            example_id=f"{domain.value}_rollback_{i+1:03d}",
            domain=domain,
            operation_type=OperationType.ROLLBACK,
            conversation_turns=[
                ConversationTurn(role="user", content="Book a flight to Paris."),
                ConversationTurn(role="assistant", content="Flight to Paris selected."),
                ConversationTurn(role="user", content="Actually, make it London."),
                ConversationTurn(role="assistant", content="Updated to London."),
                ConversationTurn(role="user", content="Nevermind, go back to the original plan (Paris).")
            ],
            initial_state={"destination": "Paris"},
            ordered_state_updates=[
                StateUpdate(operation_type=OperationType.OVERWRITE, field="destination", old_value="Paris", new_value="London"),
                StateUpdate(operation_type=OperationType.ROLLBACK)
            ],
            final_active_state=tracker.get_active_state(),
            superseded_values=tracker.get_superseded_values(),
            expected_tool_name=f"book_{domain.value}",
            expected_normalized_tool_arguments={"destination": "Paris"},
            ambiguity_status=False,
            expected_clarification_behavior=None,
            oracle_clean_turns=[
                ConversationTurn(role="user", content="Book a flight to Paris.")
            ]
        )
        examples.append(ex)

    # 5. TOOL_STATE_UPDATE
    for i in range(6):
        domain = domains[(i + 4) % len(domains)]
        tracker = StateTracker()
        
        tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="time", new_value="tomorrow"))
        tracker.apply_update(StateUpdate(operation_type=OperationType.TOOL_UPDATE, field="time", new_value="2026-07-26"))
        
        ex = PilotExample(
            example_id=f"{domain.value}_tool_update_{i+1:03d}",
            domain=domain,
            operation_type=OperationType.TOOL_STATE_UPDATE,
            conversation_turns=[
                ConversationTurn(role="user", content="Schedule an appointment for tomorrow."),
                ConversationTurn(role="assistant", content="Checking calendar for tomorrow... (Tool resolved to 2026-07-26)"),
                ConversationTurn(role="user", content="Go ahead and book it.")
            ],
            initial_state={"time": "tomorrow"},
            ordered_state_updates=[
                StateUpdate(operation_type=OperationType.TOOL_UPDATE, field="time", old_value="tomorrow", new_value="2026-07-26")
            ],
            final_active_state=tracker.get_active_state(),
            superseded_values=tracker.get_superseded_values(),
            expected_tool_name=f"confirm_{domain.value}",
            expected_normalized_tool_arguments={"time": "2026-07-26"},
            ambiguity_status=False,
            expected_clarification_behavior=None,
            oracle_clean_turns=[
                ConversationTurn(role="user", content="Schedule an appointment for 2026-07-26.")
            ]
        )
        examples.append(ex)

    return examples

def generate_report(examples: List[PilotExample], filepath: str):
    domain_counts = defaultdict(int)
    op_counts = defaultdict(int)
    
    for ex in examples:
        domain_counts[ex.domain.value] += 1
        op_counts[ex.operation_type.value] += 1
        
    lines = [
        "# Pilot Dataset Report\n",
        f"Total examples: {len(examples)}\n",
        "## By Domain",
    ]
    for d, c in domain_counts.items():
        lines.append(f"- {d}: {c}")
        
    lines.append("\n## By Operation Type")
    for o, c in op_counts.items():
        lines.append(f"- {o}: {c}")
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

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
    print(f"Generated {len(examples)} examples.")
