import copy
from typing import Dict, List, Any
from .schemas import OperationType, StateUpdate, Tombstone

class StateTracker:
    def __init__(self):
        self.active: Dict[str, Any] = {}
        self.superseded: Dict[str, List[Any]] = {}
        self.deleted: Dict[str, Tombstone] = {}
        self.cancelled: List[str] = []
        self.ambiguities: List[Dict[str, str]] = []
        self._history: List[Dict[str, Any]] = []

    def snapshot(self):
        self._history.append({
            "active": copy.deepcopy(self.active),
            "superseded": copy.deepcopy(self.superseded),
            "deleted": copy.deepcopy(self.deleted),
            "cancelled": copy.deepcopy(self.cancelled),
            "ambiguities": copy.deepcopy(self.ambiguities),
        })

    def apply_update(self, update: StateUpdate):
        self.snapshot()
        
        op = update.operation_type
        field = update.field
        
        if op in (OperationType.SET, OperationType.OVERWRITE, OperationType.REPLACE, OperationType.TOOL_UPDATE, OperationType.TOOL_STATE_UPDATE):
            if field in self.active:
                old_val = self.active[field]
                if field not in self.superseded:
                    self.superseded[field] = []
                self.superseded[field].append(old_val)
            self.active[field] = update.new_value
            if field in self.deleted:
                del self.deleted[field]

        elif op == OperationType.DELETION:
            if field in self.active:
                old_val = self.active.pop(field)
                self.deleted[field] = Tombstone(previous_value=old_val)

        elif op == OperationType.CANCELLATION:
            val_to_cancel = update.new_value or "action"
            self.cancelled.append(val_to_cancel)

        elif op == OperationType.ROLLBACK:
            if len(self._history) >= 1:
                # The snapshot we just took in apply_update
                self._history.pop() 
                
                if len(self._history) >= 1:
                    target_state = self._history[-1]
                    current_active = self.active
                    
                    self.active = copy.deepcopy(target_state["active"])
                    self.deleted = copy.deepcopy(target_state["deleted"])
                    self.cancelled = copy.deepcopy(target_state["cancelled"])
                    self.ambiguities = copy.deepcopy(target_state["ambiguities"])
                    
                    for k, v in current_active.items():
                        if k not in self.active or self.active[k] != v:
                            if k not in self.superseded:
                                self.superseded[k] = []
                            self.superseded[k].append(v)

    def get_active_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.active)

    def get_superseded_values(self) -> Dict[str, List[Any]]:
        return copy.deepcopy(self.superseded)
