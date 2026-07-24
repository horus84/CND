from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

class OperationType(str, Enum):
    SET = "set"
    REPLACE = "replace"
    OVERWRITE = "overwrite"
    DELETION = "deletion"
    CANCELLATION = "cancellation"
    ROLLBACK = "rollback"
    TOOL_UPDATE = "tool_update"
    TOOL_STATE_UPDATE = "tool_state_update"

class Domain(str, Enum):
    CALENDAR = "calendar"
    TRAVEL = "travel"
    EMAIL = "email"
    SHOPPING = "shopping"
    APPOINTMENT = "appointment"

class ConversationTurn(BaseModel):
    role: str
    content: str

class Tombstone(BaseModel):
    status: str = "deleted"
    previous_value: Any

class StateUpdate(BaseModel):
    operation_type: OperationType
    field: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

class PilotExample(BaseModel):
    example_id: str
    domain: Domain
    operation_type: OperationType
    conversation_turns: List[ConversationTurn]
    initial_state: Dict[str, Any]
    ordered_state_updates: List[StateUpdate]
    final_active_state: Dict[str, Any]
    superseded_values: Dict[str, List[Any]]
    expected_tool_name: Optional[str]
    expected_normalized_tool_arguments: Dict[str, Any]
    ambiguity_status: bool
    expected_clarification_behavior: Optional[str]
    
class ModelOutput(BaseModel):
    tool: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    clarification: Optional[str] = None
    raw_output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    strategy: str = ""
    model_revision: str = ""
    transformers_version: str = ""
    random_seed: int = 42

ALLOWED_TOOLS = [
    {
        "name": "create_calendar_event",
        "description": "Schedule a new event on the calendar.",
        "arguments": ["time", "day", "event"]
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "arguments": ["target", "topic"]
    },
    {
        "name": "book_travel",
        "description": "Book a flight or travel ticket.",
        "arguments": ["destination"]
    },
    {
        "name": "book_shopping",
        "description": "Purchase an item.",
        "arguments": ["item"]
    },
    {
        "name": "create_appointment_event",
        "description": "Schedule a new appointment.",
        "arguments": ["time", "day"]
    },
    {
        "name": "confirm_calendar",
        "description": "Confirm a calendar update.",
        "arguments": ["time", "day"]
    },
    {
        "name": "confirm_appointment",
        "description": "Confirm an appointment update.",
        "arguments": ["time", "day"]
    }
]
