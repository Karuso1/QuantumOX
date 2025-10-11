"""
engine.py

QuantumOX engine glue: connects Board, Searcher and Options into a single
Engine class suitable for use by main.py (the UTTTI loop).

Responsibilities
- Manage the current Board instance and apply incoming moves
- Expose set_option / new_game / play_move / go functionality
- Convert Searcher results into UTTTI-friendly info lines and bestmove line

Author: Kartik (QuantumOX project)
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

from constants import DEFAULT_GRID
from board import Board
from search import Searcher
from options import set_option, get_option
from utils import (
    format_info_line,
    format_info_string,
    format_bestmove,
)


class QuantumOXEngine:
    def __init__(self) -> None:
        # create board according to current option (if options module not yet configured,
        # fallback to DEFAULT_GRID)
        try:
            grid = get_option("Grid")
        except Exception:
            grid = DEFAULT_GRID
        self.board = Board(grid_spec=grid)
        self.searcher = Searcher()

    # ------------------ options / game lifecycle ---------------------------
    def set_option(self, name: str, value: str) -> Tuple[bool, str]:
        """Set option and perform any side-effects (e.g. changing grid resets board).

        Returns (success, info_message) where info_message is already the text to
        be used in an `info string` UTTTI reply.
        """
        success, msg = set_option(name, value)
        # if Grid changed, recreate board
        if success and name == "Grid":
            # recreate board with new grid spec
            self.board = Board(grid_spec=value)
            # reflect new starting player from options
            try:
                self.board.side_to_move = get_option("FirstPlayer")
            except Exception:
                pass
        return success, msg

    def new_game(self) -> None:
        """Reset board to a fresh game silently (no info string)."""
        self.board = Board(grid_spec=get_option("Grid"))
        try:
            self.board.side_to_move = get_option("FirstPlayer")
        except Exception:
            pass

    def play_moves(self, moves: List[int]) -> str:
        """Apply a sequence of moves to the board (used for `grid ... fill` style commands)."""
        for mv in moves:
            self.board.make_move(mv)
        return format_info_string(f"applied {len(moves)} moves")

    def play_move(self, move: int) -> str:
        """Apply a single move (1-based). Returns an info string confirming the move."""
        self.board.make_move(move)
        return format_info_string(f"played {move}")

    # ------------------ searching / go -----------------------------------
    def go(self, depth: Optional[int] = None, time_ms: Optional[int] = None, nodes: Optional[int] = None) -> Dict[str, Any]:
        """Run search and return a dict with:
            - info_lines: List[str] (UTTTI `info` lines)
            - bestmove_line: str (the final `bestmove ...` line)
            - bestmove: Optional[int]
            - raw: dict (raw search output from Searcher)
        """
        res = self.searcher.search(self.board, max_depth=depth, time_ms=time_ms, nodes_limit=nodes)
        info_lines: List[str] = []
        for d in res.get("infos", []):
            # d contains keys: depth, seldepth, score, nodes, pv
            line = format_info_line(
                depth=d.get("depth"),
                seldepth=d.get("seldepth"),
                score_cp=d.get("score"),
                nodes=d.get("nodes"),
                pv=d.get("pv"),
            )
            info_lines.append(line)

        bestmove = res.get("bestmove")
        # attempt to set a ponder move to the second PV move if present
        pv = res.get("pv", [])
        ponder = pv[1] if len(pv) > 1 else None
        if bestmove is None:
            bestmove_line = format_bestmove(0, ponder=None)
        else:
            bestmove_line = format_bestmove(bestmove, ponder=ponder)

        return {
            "info_lines": info_lines,
            "bestmove_line": bestmove_line,
            "bestmove": bestmove,
            "raw": res,
        }
    
    def stop(self):
        if hasattr(self, "searcher"):
            self.searcher.abort = True

    # ------------------ utilities ----------------------------------------
    def board_state(self) -> str:
        return str(self.board)

    def legal_moves(self) -> List[int]:
        return self.board.legal_moves()

__all__ = ["QuantumOXEngine"]
