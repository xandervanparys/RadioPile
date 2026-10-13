"""
Label registry utilities.

Responsibilities:
- Assign stable integer IDs to labels
- Maintain label ↔ ID mappings
"""

from typing import Dict, List

LABEL_TO_ID: Dict[str, int] = {}
ID_TO_LABEL: List[str] = []


def reset_label_registry():
    LABEL_TO_ID.clear()
    ID_TO_LABEL.clear()


def get_or_create_label_id(label: str) -> int:
    if label in LABEL_TO_ID:
        return LABEL_TO_ID[label]
    idx = len(ID_TO_LABEL)
    LABEL_TO_ID[label] = idx
    ID_TO_LABEL.append(label)
    return idx

def get_label_names() -> List[str]:
    return ID_TO_LABEL.copy()
