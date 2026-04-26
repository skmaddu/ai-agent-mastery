# Demonstrating state with custom reducers

from typing import Annotated, TypedDict
from datetime import datetime

# Reducers
def min_reducer(existing: int, new: int) -> int:
    return min(existing, new)

def max_reducer(existing: int, new: int) -> int:
    return max(existing, new)

def datetime_max_reducer(existing: datetime, new: datetime) -> datetime:
    return max(existing, new)

def dict_merge_reducer(existing: dict[str, str], new: dict[str, str]) -> dict[str, str]:
    return {**existing, **new}

def set_union_reducer(existing: set[int], new: set[int]) -> set[int]:
    return existing | new

# Workflow state definition
class WorkflowState(TypedDict):
    min_cost: Annotated[int, min_reducer]
    max_score: Annotated[int, max_reducer]
    latest_update: Annotated[datetime, datetime_max_reducer]
    metadata: Annotated[dict[str, str], dict_merge_reducer]
    unique_ids: Annotated[set[int], set_union_reducer]