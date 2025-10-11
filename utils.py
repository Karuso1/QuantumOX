"""
utils.py

Helper utilities for QuantumOX UTTTI engine.

Contains parsing helpers, move/coordinate conversion, simple formatting helpers
for UTTTI "info" and "bestmove" lines, and small board pretty-printer.

Keep this file lightweight so main.py and other modules can import it freely.

Author: Kartik
"""
from __future__ import annotations

import shlex
from typing import List, Tuple, Optional, Iterable

from constants import (
    PV_SEPARATOR,
    INFO_STRING_PREFIX,
    SYMBOL_EMPTY,
    DEFAULT_GRID,
    parse_grid_spec,
)


# --- command tokenization ----------------------------------------------------
def tokenize_command(line: str) -> List[str]:
    """Split a UTTTI command line into tokens while respecting quoted strings.

    Example:
        tokenize_command('info string set "Grid" to 3x3')
        -> ['info', 'string', 'set', 'Grid', 'to', '3x3']
    """
    try:
        return shlex.split(line)
    except ValueError:
        # Fall back to simple whitespace split if shlex fails for some reason.
        return line.strip().split()


# --- setoption parsing ------------------------------------------------------
def parse_setoption(tokens: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Parse tokens from a `setoption` command.

    Expected style: setoption name <NAME> value <VALUE>
    Returns (name, value) or (None, None) if parsing fails.
    """
    if not tokens:
        return None, None
    # tokens typically start with ['setoption', 'name', 'Foo', 'value', 'Bar']
    # we'll search for the keywords name/value and pull the next token.
    name = None
    value = None
    for i, t in enumerate(tokens):
        if t.lower() == "name" and i + 1 < len(tokens):
            name = tokens[i + 1]
        if t.lower() == "value" and i + 1 < len(tokens):
            value = tokens[i + 1]
    return name, value


# --- grid helpers -----------------------------------------------------------
def default_grid_dims() -> Tuple[int, ...]:
    return parse_grid_spec(DEFAULT_GRID)


def index_to_coords(index: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert 1-based linear index to multidimensional coordinates.

    - `index` is 1-based (as in many UTTTI examples where moves are 1..N)
    - `dims` is a tuple, e.g. (3,3) or (3,3,3)

    Returns a tuple of coordinates (zero-based) matching dims order.
    """
    if index < 1:
        raise ValueError("Index must be >= 1")
    idx = index - 1
    coords: List[int] = []
    # row-major conversion for arbitrary dimensions
    for size in reversed(dims):
        coords.append(idx % size)
        idx //= size
    if idx != 0:
        raise ValueError("Index out of range for grid dimensions")
    return tuple(reversed(coords))


def coords_to_index(coords: Iterable[int], dims: Tuple[int, ...]) -> int:
    """Convert coordinates (zero-based) to 1-based linear index.

    Example: (0,0) on a 3x3 -> 1, (1,1) -> 5
    """
    coords = tuple(coords)
    if len(coords) != len(dims):
        raise ValueError("coords length must match dims length")
    idx = 0
    for c, size in zip(coords, dims):
        if c < 0 or c >= size:
            raise ValueError("coordinate out of range")
        idx = idx * size + c
    return idx + 1


# --- move parsing -----------------------------------------------------------
def parse_move_token(tok: str) -> int:
    """Parse a move token into a 1-based integer index.

    Currently supports:
      - plain integers like '5' -> 5
      - future-proof: grid coords like '1,2' (row,col) -> converted to index
    """
    tok = tok.strip()
    # plain integer
    if tok.isdigit():
        return int(tok)
    # comma-separated coords
    if "," in tok:
        parts = tok.split(",")
        coords = tuple(int(p) for p in parts)
        dims = default_grid_dims()
        return coords_to_index(coords, dims)
    raise ValueError(f"Unrecognized move token: {tok!r}")


# --- formatting helpers for UTTTI output -----------------------------------
def format_info_line(
    depth: Optional[int] = None,
    seldepth: Optional[int] = None,
    score_cp: Optional[int] = None,
    nodes: Optional[int] = None,
    pv: Optional[List[int]] = None,
) -> str:
    parts: List[str] = ["info"]
    if depth is not None:
        parts += ["depth", str(depth)]
    if seldepth is not None:
        parts += ["seldepth", str(seldepth)]
    if score_cp is not None:
        parts += ["score", "cp", str(score_cp)]
    if nodes is not None:
        parts += ["nodes", str(nodes)]
    if pv:
        parts += ["pv"] + [str(m) for m in pv]
    return " ".join(parts)


def format_info_string(msg: str) -> str:
    return f"{INFO_STRING_PREFIX} {msg}"


def format_bestmove(move: int, ponder: Optional[int] = None) -> str:
    if ponder is None:
        return f"bestmove {move}"
    return f"bestmove {move} ponder {ponder}"


# --- board pretty-printer ---------------------------------------------------
def pretty_print_board(cells: List[str], dims: Tuple[int, ...]) -> str:
    """Return a human-friendly string representation for 2D boards.

    For 1D or >2D boards we fallback to a compact linear representation.
    - `cells` should be a flat list of length product(dims).
    """
    if len(dims) == 2:
        rows, cols = dims
        if len(cells) != rows * cols:
            raise ValueError("cells length doesn't match dims")
        lines: List[str] = []
        for r in range(rows):
            line = " ".join(cells[r * cols : (r + 1) * cols])
            lines.append(line)
        return "\n".join(lines)
    # fallback: linear with indices
    return " ".join(f"{i+1}:{c}" for i, c in enumerate(cells))


# --- tiny helpers -----------------------------------------------------------
def empty_grid_cells(dims: Tuple[int, ...]) -> List[str]:
    total = 1
    for d in dims:
        total *= d
    return [SYMBOL_EMPTY] * total


# Exports
__all__ = [
    "tokenize_command",
    "parse_setoption",
    "index_to_coords",
    "coords_to_index",
    "parse_move_token",
    "format_info_line",
    "format_info_string",
    "format_bestmove",
    "pretty_print_board",
    "empty_grid_cells",
]
