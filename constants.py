"""
constants.py

Static constants and tiny helpers for the UTTTI Universal Tic-Tac-Toe Interface.
This file intentionally stays small and stable — other modules import these values.

Author: Kartik
"""
from __future__ import annotations

from typing import Tuple, Union

# Engine identity -------------------------------------------------------------
ENGINE_NAME = "QuantumOX"
ENGINE_AUTHOR = "Kartik"
UTTTI_VERSION = "0.1"

# Default options ----------------------------------------------------------------
DEFAULT_GRID = "3x3"  # common canonical default
SUPPORTED_GRIDS = (
    "3x3",
    "4x4",
    "5x5",
    "3x3x3",  # simple 3D example string (layers x rows x cols)
)

# Player/square symbols ------------------------------------------------------
# Keep these simple strings so UI and board modules can format however they want.
SYMBOL_EMPTY = "."
SYMBOL_X = "X"
SYMBOL_O = "O"

# Search score sentinels -----------------------------------------------------
SCORE_MATE = 10_000_000
SCORE_WIN = 1_000_000
SCORE_DRAW = 0
SCORE_LOSS = -SCORE_WIN

# UTTTI protocol keywords / tokens -------------------------------------------
CMD_UTTTI = "uttti"
CMD_ID = "id"
CMD_ISREADY = "isready"
CMD_READYOK = "readyok"
CMD_SETOPTION = "setoption"
CMD_INFO = "info"
CMD_NEWGAME = "utttinewgame"
CMD_GRID = "grid"
CMD_GO = "go"
CMD_BESTMOVE = "bestmove"

# Info keys the engine will emit (keeps formatting consistent across modules)
INFO_KEYS = ("depth", "seldepth", "score", "nodes", "pv")

# Search limit keywords accepted by `go` command
SEARCH_LIMITS = ("depth", "movetime", "nodes", "infinite")

# Default search settings ----------------------------------------------------
DEFAULT_MAX_DEPTH = 6
DEFAULT_TIME_MS = 1000

# Misc / formatting ----------------------------------------------------------
PV_SEPARATOR = " "
INFO_STRING_PREFIX = "info string"

# Public API -----------------------------------------------------------------
__all__ = [
    "ENGINE_NAME",
    "ENGINE_AUTHOR",
    "UTTTI_VERSION",
    "DEFAULT_GRID",
    "SUPPORTED_GRIDS",
    "SYMBOL_EMPTY",
    "SYMBOL_X",
    "SYMBOL_O",
    "SCORE_MATE",
    "SCORE_WIN",
    "SCORE_DRAW",
    "SCORE_LOSS",
    "CMD_UTTTI",
    "CMD_ID",
    "CMD_ISREADY",
    "CMD_READYOK",
    "CMD_SETOPTION",
    "CMD_INFO",
    "CMD_NEWGAME",
    "CMD_GRID",
    "CMD_GO",
    "CMD_BESTMOVE",
    "INFO_KEYS",
    "SEARCH_LIMITS",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_TIME_MS",
    "PV_SEPARATOR",
    "INFO_STRING_PREFIX",
]


# --- tiny helper: parse grid spec -------------------------------------------
def parse_grid_spec(spec: str) -> Tuple[int, ...]:
    """Parse grid spec strings like "3x3" or "3x3x3" into ints (rows, cols, ...).

    Examples
    --------
    >>> parse_grid_spec("3x3")
    (3, 3)

    >>> parse_grid_spec("3x3x3")
    (3, 3, 3)
    """
    parts = spec.split("x")
    try:
        nums = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(f"Invalid grid spec: {spec!r}. Expected format like '3x3' or '3x3x3'.")
    return nums


# Backwards-compatible alias -------------------------------------------------
def grid_dims(spec: Union[str, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Return grid dimensions as a tuple. If already a tuple, return it unchanged."""
    if isinstance(spec, tuple):
        return spec
    return parse_grid_spec(spec)
