import pytest
from src.schemas import OperationType, StateUpdate, Tombstone
from src.state_tracker import StateTracker

def test_overwrite_removes_previous_active_value():
    tracker = StateTracker()
    tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="hotel", new_value="Grand Hotel"))
    assert tracker.active["hotel"] == "Grand Hotel"
    assert "hotel" not in tracker.superseded

    tracker.apply_update(StateUpdate(operation_type=OperationType.OVERWRITE, field="hotel", new_value="Hilton"))
    assert tracker.active["hotel"] == "Hilton"
    assert tracker.superseded["hotel"] == ["Grand Hotel"]

def test_deletion_creates_tombstone():
    tracker = StateTracker()
    tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="flight", new_value="AA123"))
    tracker.apply_update(StateUpdate(operation_type=OperationType.DELETION, field="flight"))
    
    assert "flight" not in tracker.active
    assert tracker.deleted["flight"].previous_value == "AA123"
    assert tracker.deleted["flight"].status == "deleted"

def test_cancellation_prevents_execution():
    tracker = StateTracker()
    tracker.apply_update(StateUpdate(operation_type=OperationType.CANCELLATION, new_value="book_flight"))
    assert "book_flight" in tracker.cancelled

def test_rollback_restores_correct_previous_state():
    tracker = StateTracker()
    tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="date", new_value="Friday"))
    tracker.apply_update(StateUpdate(operation_type=OperationType.OVERWRITE, field="date", new_value="Saturday"))
    assert tracker.active["date"] == "Saturday"
    
    tracker.apply_update(StateUpdate(operation_type=OperationType.ROLLBACK))
    assert tracker.active["date"] == "Friday"
    assert "Saturday" in tracker.superseded["date"]

def test_tool_updates_override_stale_conversational_values():
    tracker = StateTracker()
    tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="time", new_value="tomorrow morning"))
    tracker.apply_update(StateUpdate(operation_type=OperationType.TOOL_UPDATE, field="time", new_value="09:00"))
    
    assert tracker.active["time"] == "09:00"
    assert tracker.superseded["time"] == ["tomorrow morning"]

def test_stale_value_detection():
    tracker = StateTracker()
    tracker.apply_update(StateUpdate(operation_type=OperationType.SET, field="name", new_value="John"))
    tracker.apply_update(StateUpdate(operation_type=OperationType.OVERWRITE, field="name", new_value="Jane"))
    tracker.apply_update(StateUpdate(operation_type=OperationType.OVERWRITE, field="name", new_value="Doe"))
    
    stale = tracker.get_superseded_values()
    assert stale["name"] == ["John", "Jane"]
