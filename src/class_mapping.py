"""
Map Keras softmax indices to GTSRB class ids (0–42).

flow_from_directory() defaults to alphabetical subfolder order, so index 2 is
class folder "10", not "2". Training writes outputs/class_order.json when
classes are forced to numeric order; without that file, we assume the legacy
alphabetical mapping so old checkpoints still decode to correct sign names.
"""

from __future__ import annotations

import json
from pathlib import Path

NUM_CLASSES = 43

# Keras alphabetical order of folder names "0".."42"
_LEGACY_INDEX_TO_GTSRB: tuple[int, ...] = tuple(
    int(x) for x in sorted(str(i) for i in range(NUM_CLASSES))
)

_decoder: tuple[int, ...] | None = None


def _load_decoder() -> tuple[int, ...]:
    global _decoder
    if _decoder is not None:
        return _decoder
    meta = Path("outputs/class_order.json")
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            if data.get("indices_are_gtsrb_ids"):
                _decoder = tuple(range(NUM_CLASSES))
                return _decoder
        except (json.JSONDecodeError, OSError):
            pass
    _decoder = _LEGACY_INDEX_TO_GTSRB
    return _decoder


def decode_prediction_index(model_index: int) -> int:
    """Turn model output class index into GTSRB label 0..42."""
    d = _load_decoder()
    i = int(model_index)
    if i < 0 or i >= len(d):
        raise IndexError(f"Class index {i} out of range for {NUM_CLASSES} classes")
    return d[i]


def reset_decoder_cache() -> None:
    """For tests or reloading after retraining."""
    global _decoder
    _decoder = None
