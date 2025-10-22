"""
board.py

Board implementation for QuantumOX (UTTTI). Provides the Board class which
implements the interface expected by search.py:

    - legal_moves() -> list[int]
    - make_move(move: int) -> None
    - unmake_move(move: int) -> None
    - side_to_move -> str
    - is_win(player: str) -> bool
    - is_draw() -> bool
    - evaluate(player: str) -> int
    - zobrist_key() -> int (optional but implemented)

Features
--------
- Supports N-dimensional cubic boards like '3x3' or '3x3x3' (and rectangular 2D like '4x3').
- Moves are 1-based indices to match UTTTI examples.
- make/unmake are LIFO and safe; unmake_move will assert LIFO usage.
- Generic win-line generator for any N-dimension board. For win length we use:
    - if all dims equal -> win_length = dims[0]
    - else -> win_length = min(dims)
- Simple heuristic evaluate(player): counts "open" lines and scores them.

Author: Kartik (QuantumOX project)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Optional
import itertools
import random

from constants import (
    SYMBOL_EMPTY,
    SYMBOL_X,
    SYMBOL_O,
    parse_grid_spec,
)


def product(nums: Iterable[int]) -> int:
    p = 1
    for n in nums:
        p *= n
    return p


@dataclass
class Board:
    grid_spec: str = "3x3"
    cells: List[str] = field(init=False)
    dims: Tuple[int, ...] = field(init=False)
    side_to_move: str = field(default=SYMBOL_X)
    move_stack: List[int] = field(default_factory=list)
    _zobrist_table: Optional[List[List[int]]] = field(init=False, default=None)

    def __post_init__(self):
        self.dims = parse_grid_spec(self.grid_spec)
        total = product(self.dims)
        self.cells = [SYMBOL_EMPTY] * total
        # deterministically seeded zobrist table per instance
        self._init_zobrist(seed=0)
        # precompute all winning lines (list of lists of 1-based indices)
        self._win_lines = list(self._generate_win_lines())

    # ------------------ move / state management ------------------------------
    def legal_moves(self) -> List[int]:
        return [i + 1 for i, c in enumerate(self.cells) if c == SYMBOL_EMPTY]

    def make_move(self, move: int) -> None:
        idx = move - 1
        if idx < 0 or idx >= len(self.cells):
            raise ValueError(f"Move {move} out of range for board")
        if self.cells[idx] != SYMBOL_EMPTY:
            raise ValueError(f"Cell {move} is not empty")
        self.cells[idx] = self.side_to_move
        self.move_stack.append(move)
        # flip side
        self.side_to_move = SYMBOL_O if self.side_to_move == SYMBOL_X else SYMBOL_X

    def unmake_move(self, move: int) -> None:
        if not self.move_stack:
            raise ValueError("Unmake called but move stack is empty")
        last = self.move_stack.pop()
        if last != move:
            # this is a safety check: search should unmake LIFO
            raise AssertionError(f"Unmake called with {move} but last move was {last}")
        idx = move - 1
        self.cells[idx] = SYMBOL_EMPTY
        # flip side back
        self.side_to_move = SYMBOL_O if self.side_to_move == SYMBOL_X else SYMBOL_X

    # ------------------ game status ----------------------------------------
    def is_win(self, player: str) -> bool:
        for line in self._win_lines:
            if all(self.cells[i - 1] == player for i in line):
                return True
        return False

    def is_draw(self) -> bool:
        if self.is_win(SYMBOL_X) or self.is_win(SYMBOL_O):
            return False
        return all(c != SYMBOL_EMPTY for c in self.cells)

    # ------------------ evaluation -----------------------------------------
    def evaluate(self, player: str) -> int:
        """Simple heuristic: score = (#player-open-lines - #opp-open-lines) * factor

        An "open" line is a winning line that contains only player marks and empty cells.
        The contribution scales with how many marks the player already has in the line.
        """
        opp = SYMBOL_O if player == SYMBOL_X else SYMBOL_X
        score = 0
        for line in self._win_lines:
            marks = [self.cells[i - 1] for i in line]
            if opp in marks and player in marks:
                continue
            if opp not in marks and player not in marks:
                # empty line: small value
                score += 1
            elif player in marks and opp not in marks:
                cnt = marks.count(player)
                score += (cnt * cnt) * 10
            elif opp in marks and player not in marks:
                cnt = marks.count(opp)
                score -= (cnt * cnt) * 10
        return score

    # ------------------ zobrist hashing ------------------------------------
    def _init_zobrist(self, seed: int = 0) -> None:
        rnd = random.Random(seed)
        # two players X/O per cell
        total = len(self.cells)
        self._zobrist_table = [ [rnd.getrandbits(64) for _ in range(2)] for __ in range(total) ]

    def zobrist_key(self) -> int:
        if self._zobrist_table is None:
            self._init_zobrist(seed=0)
        h = 0
        for i, c in enumerate(self.cells):
            if c == SYMBOL_X:
                h ^= self._zobrist_table[i][0]
            elif c == SYMBOL_O:
                h ^= self._zobrist_table[i][1]
        # include side to move bit
        if self.side_to_move == SYMBOL_O:
            h ^= 0xF00DF00DCAFEBABE
        return h

    # ------------------ win-line generation -------------------------------
    def _generate_win_lines(self) -> Iterable[List[int]]:
        """Yield all straight lines of length `L` in the N-dimensional grid.

        Strategy:
        - win_length = dims[0] if all dims equal, else min(dims)
        - directions: all vectors in {-1,0,1}^N excluding all-zero; keep a canonical
          orientation to avoid duplicate reverse directions.
        - for each start cell and direction, if the full line of length L fits in bounds,
          yield the list of 1-based linear indices.
        """
        N = len(self.dims)
        if N == 0:
            return
        if all(d == self.dims[0] for d in self.dims):
            L = self.dims[0]
        else:
            L = min(self.dims)
        ranges = [range(d) for d in self.dims]

        # direction vectors
        directions = list(itertools.product([-1, 0, 1], repeat=N))
        # remove zero vector
        directions = [d for d in directions if any(x != 0 for x in d)]
        # canonicalize: keep only directions where the first non-zero component is positive
        def canonical(d):
            for x in d:
                if x != 0:
                    return x > 0
            return False
        directions = [d for d in directions if canonical(d)]

        # function to test if a line fits starting at coords and direction
        def in_bounds(coords):
            return all(0 <= c < self.dims[i] for i, c in enumerate(coords))

        # iterate all starts and directions
        for start in itertools.product(*ranges):
            for d in directions:
                line_coords = [tuple(start[i] + k * d[i] for i in range(N)) for k in range(L)]
                if all(in_bounds(c) for c in line_coords):
                    # convert coords -> 1-based index
                    inds = [self._coords_to_index(c) for c in line_coords]
                    yield inds

    def _coords_to_index(self, coords: Tuple[int, ...]) -> int:
        # row-major
        idx = 0
        for c, size in zip(coords, self.dims):
            idx = idx * size + c
        return idx + 1

    # ------------------ utilities -----------------------------------------
    def fill_from_list(self, moves: Iterable[int]) -> None:
        """Fill the board by applying moves in sequence (useful for starting positions).

        Moves are played in the order given; side_to_move alternates starting from current.
        """
        for mv in moves:
            self.make_move(mv)

    def reset(self) -> None:
        self.cells = [SYMBOL_EMPTY] * len(self.cells)
        self.move_stack.clear()
        self.side_to_move = SYMBOL_X

    def __str__(self) -> str:
        # pretty-print 2D boards, else linear
        if len(self.dims) == 2:
            rows, cols = self.dims
            lines = []
            for r in range(rows):
                lines.append(" ".join(self.cells[r*cols:(r+1)*cols]))
            return "\n".join(lines)
        return " ".join(f"{i+1}:{c}" for i, c in enumerate(self.cells))


__all__ = [
    "Board",
]
