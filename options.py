"""
options.py

Engine options manager for QuantumOX UTTTI engine.
Provides a small, explicit options registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from constants import (
    DEFAULT_GRID,
    SUPPORTED_GRIDS,
    SYMBOL_X,
    SYMBOL_O,
    parse_grid_spec,
)

@dataclass
class Option:
    name: str
    type: str  # 'string', 'bool', etc.
    default: Any
    value: Any = field(default=None)
    description: str = ""
    validator: Optional[Callable[[Any], Any]] = None

    def __post_init__(self):
        if self.value is None:
            self.value = self.default

    def set(self, raw_value: Any) -> None:
        if self.type == "bool":
            sv = str(raw_value).lower()
            if sv in ("true", "1", "yes", "on"):
                v = True
            elif sv in ("false", "0", "no", "off"):
                v = False
            else:
                raise ValueError(f"Option {self.name} expects a bool-like value")
        else:
            v = str(raw_value)

        if self.validator:
            v = self.validator(v)

        self.value = v

# --- validators ---

def validate_grid(v: str) -> str:
    if v not in SUPPORTED_GRIDS:
        raise ValueError(f"Unsupported grid '{v}'. Supported: {', '.join(SUPPORTED_GRIDS)}")
    _ = parse_grid_spec(v)
    return v


def validate_firstplayer(v: str) -> str:
    sv = str(v).upper()
    if sv not in (SYMBOL_X, SYMBOL_O):
        raise ValueError("FirstPlayer must be 'X' or 'O'")
    return sv

# --- default registry ---
_registry: Dict[str, Option] = {
    "Grid": Option(name="Grid", type="string", default=DEFAULT_GRID, description="Board grid specification, e.g. '3x3', '4x4', or '3x3x3'.", validator=validate_grid),
    "FirstPlayer": Option(name="FirstPlayer", type="string", default=SYMBOL_X, description="Symbol for the player who moves first: 'X' or 'O'.", validator=validate_firstplayer),
}

# --- public API ---

def set_option(name: str, raw_value: Any) -> Tuple[bool, str]:
    opt = _registry.get(name)
    if not opt:
        return False, f"Unknown option '{name}'"
    try:
        opt.set(raw_value)
    except Exception as e:
        return False, f"Failed to set option '{name}': {e}"
    return True, f'set "{name}" to {opt.value}'


def get_option(name: str) -> Any:
    opt = _registry.get(name)
    if not opt:
        raise KeyError(f"Unknown option '{name}'")
    return opt.value


def list_options() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, opt in _registry.items():
        out[k] = {
            "type": opt.type,
            "default": opt.default,
            "value": opt.value,
            "description": opt.description,
        }
    return out


def get_grid_dims() -> Tuple[int, ...]:
    grid_spec = get_option("Grid")
    return parse_grid_spec(grid_spec)

__all__ = [
    "set_option", 
    "get_option", 
    "list_options", 
    "get_grid_dims", 
    "Option"
]
