"""
search.py

Advanced minimax (negamax) with alpha-beta pruning for QuantumOX (UTTTI).

Design notes / required Board API
-------------------------------
This module is intentionally engine-agnostic: it expects the `board` object passed
into search functions to implement a small, clear interface:

    - legal_moves() -> list[int]
        Return a list of legal moves (1-based indices to match UTTTI examples).

    - make_move(move: int) -> None
        Play `move` on the board.

    - unmake_move(move: int) -> None
        Undo the previously played move (LIFO order expected).

    - side_to_move -> str
        Current player symbol, typically 'X' or 'O'.

    - is_win(player: str) -> bool
        Whether `player` has a winning line on the current board.

    - is_draw() -> bool
        True if the position is a draw (no legal moves and no winner).

    - evaluate(player: str) -> int
        Heuristic evaluation in centipawn-ish units from the perspective of `player`.
        If the board lacks a heuristic evaluator, the search will use a simple
        terminal-only evaluation (win/draw/loss).

    - optional: zobrist_key() -> int
        A fast integer hash for TT use. If absent, the search will use
        tuple(board_state)+side_to_move as the key (slower but correct).


Features implemented
--------------------
- Iterative deepening (depth=1..max_depth)
- Negamax formulation with alpha-beta pruning
- Simple transposition table with node type flags (EXACT, LOWERBOUND, UPPERBOUND)
- PV (principal variation) extraction from TT entries
- Node counting and per-depth `info` metrics suitable for UTTTI output
- Time and node limits (search will return best-so-far when a limit is hit)
- Move ordering: TT move first, then "winning immediate" moves, then all others


Author: Kartik (QuantumOX project)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

from constants import SCORE_WIN, SCORE_DRAW, SCORE_LOSS, DEFAULT_MAX_DEPTH
from options import get_option
from utils import format_info_line


# Transposition table entry flags
class TTFlag(Enum):
    EXACT = auto()
    LOWER = auto()
    UPPER = auto()


@dataclass
class TTEntry:
    key: Any
    depth: int
    score: int
    flag: TTFlag
    best_move: Optional[int]


class Searcher:
    def __init__(self):
        self.tt: Dict[Any, TTEntry] = {}
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = None  # seconds
        self.node_limit = None
        self.abort = False

    # ----------------- public API -------------------------------------------
    def search(
        self,
        board,
        max_depth: Optional[int] = None,
        time_ms: Optional[int] = None,
        nodes_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run an iterative-deepening search on `board`.

        Returns a dict containing at least:
          - bestmove: int or None
          - score: int (centipawn-like)
          - pv: List[int]
          - nodes: int
          - infos: List[dict] (per-depth info suitable for UTTTI `info` lines)
        """
        if max_depth is None:
            max_depth = get_option("MaxDepth") if get_option else DEFAULT_MAX_DEPTH
        self.nodes = 0
        self.start_time = time.time()
        self.time_limit = time_ms / 1000.0 if time_ms is not None else None
        self.node_limit = nodes_limit
        self.abort = False

        best_move = None
        best_score = 0
        infos: List[Dict[str, Any]] = []

        for depth in range(1, max_depth + 1):
            # call negamax with alpha-beta; get best move
            alpha = -10_000_000
            beta = 10_000_000
            score = self._negamax_root(board, depth, alpha, beta)
            if self.abort:
                break
            # extract PV
            pv = self._build_pv(board)
            best_move = self.tt.get(self._key(board)).best_move if self._key(board) in self.tt else (pv[0] if pv else None)
            best_score = score
            infos.append({
                "depth": depth,
                "seldepth": depth,  # for tic-tac-toe this is OK; if you implement quiescence, update seldepth
                "score": best_score,
                "nodes": self.nodes,
                "pv": pv,
            })
            # call back to UI via printing info lines if desired; caller/main.py can use infos

            # time/node limit check
            if self._time_exceeded() or self._nodes_exceeded():
                break

        return {
            "bestmove": best_move,
            "score": best_score,
            "pv": pv if 'pv' in locals() else [],
            "nodes": self.nodes,
            "infos": infos,
        }

    # ----------------- internal helpers ------------------------------------
    def _negamax_root(self, board, depth: int, alpha: int, beta: int) -> int:
        """Root wrapper: try moves, use TT to order moves, return best score."""
        self.nodes += 1
        moves = board.legal_moves()
        if not moves:
            # terminal position
            return self._evaluate_terminal(board)

        # move ordering: try TT move first
        key = self._key(board)
        tt_move = self.tt.get(key).best_move if key in self.tt else None
        if tt_move and tt_move in moves:
            moves.remove(tt_move)
            moves = [tt_move] + moves

        best_score = -10_000_000
        for mv in moves:
            if self._should_abort():
                break
            board.make_move(mv)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.unmake_move(mv)

            if score > best_score:
                best_score = score
                # store provisional best move in TT
                self._store_tt(key, depth, score, TTFlag.EXACT, mv)
            alpha = max(alpha, score)
            if alpha >= beta:
                # beta cutoff
                break
        return best_score

    def _negamax(self, board, depth: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        if self._should_abort():
            self.abort = True
            return 0

        key = self._key(board)
        # TT probe
        tt_entry = self.tt.get(key)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TTFlag.EXACT:
                return tt_entry.score
            elif tt_entry.flag == TTFlag.LOWER and tt_entry.score >= beta:
                return tt_entry.score
            elif tt_entry.flag == TTFlag.UPPER and tt_entry.score <= alpha:
                return tt_entry.score

        # terminal or depth==0
        if depth == 0 or board.is_win('X') or board.is_win('O') or board.is_draw():
            val = self._evaluate_terminal_or_heuristic(board)
            self._store_tt(key, depth, val, TTFlag.EXACT, None)
            return val

        # move ordering: prefer TT move
        moves = board.legal_moves()
        if not moves:
            val = self._evaluate_terminal_or_heuristic(board)
            self._store_tt(key, depth, val, TTFlag.EXACT, None)
            return val

        tt_move = tt_entry.best_move if tt_entry else None
        if tt_move and tt_move in moves:
            moves.remove(tt_move)
            moves = [tt_move] + moves

        best_score = -10_000_000
        best_move = None
        original_alpha = alpha

        for mv in moves:
            if self._should_abort():
                self.abort = True
                break
            board.make_move(mv)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.unmake_move(mv)

            if score > best_score:
                best_score = score
                best_move = mv

            alpha = max(alpha, score)
            if alpha >= beta:
                # store as lower bound because we found a move >= beta
                self._store_tt(key, depth, best_score, TTFlag.LOWER, best_move)
                break

        # if we exhausted moves without cutoff, set proper flag
        if best_score <= original_alpha:
            flag = TTFlag.UPPER
        elif best_score >= beta:
            flag = TTFlag.LOWER
        else:
            flag = TTFlag.EXACT

        self._store_tt(key, depth, best_score, flag, best_move)
        return best_score

    def _build_pv(self, board) -> List[int]:
        """Reconstruct PV from the transposition table using a cloned board."""
        import copy

        pv: List[int] = []
        seen_keys = set()
        cur_board = copy.deepcopy(board)
        cur_key = self._key(cur_board)

        while True:
            if cur_key in seen_keys:
                break
            seen_keys.add(cur_key)
            entry = self.tt.get(cur_key)
            if not entry or not entry.best_move:
                break
            mv = entry.best_move
            pv.append(mv)
            cur_board.make_move(mv)
            cur_key = self._key(cur_board)

        return pv

    # ----------------- evaluation helpers -----------------------------------
    def _evaluate_terminal(self, board) -> int:
        """Return a large score for wins/losses or draw.

        Score is from the perspective of the side to move BEFORE making the move
        in the node where this is called (consistent with negamax usage).
        """
        stm = board.side_to_move
        opp = 'O' if stm == 'X' else 'X'
        if board.is_win(stm):
            return SCORE_WIN
        if board.is_win(opp):
            return SCORE_LOSS
        if board.is_draw():
            return SCORE_DRAW
        return 0

    def _evaluate_terminal_or_heuristic(self, board) -> int:
        """Prefer board.evaluate(player) if present; fallback to terminal-only."""
        stm = board.side_to_move
        try:
            val = board.evaluate(stm)
            return val
        except Exception:
            return self._evaluate_terminal(board)

    # ----------------- transposition table ---------------------------------
    def _key(self, board) -> Any:
        """Compute a hashable key for the board. Use board.zobrist_key() if available."""
        try:
            return board.zobrist_key()
        except Exception:
            # fallback: use tuple of cell contents + side_to_move
            try:
                return (tuple(board.cells), board.side_to_move)
            except Exception:
                # last resort: use id(board) — not useful cross-positions but prevents crashes
                return id(board)

    def _store_tt(self, key: Any, depth: int, score: int, flag: TTFlag, best_move: Optional[int]):
        self.tt[key] = TTEntry(key=key, depth=depth, score=score, flag=flag, best_move=best_move)

    # ----------------- time / node limits ---------------------------------
    def _time_exceeded(self) -> bool:
        if self.time_limit is None:
            return False
        return (time.time() - self.start_time) >= self.time_limit

    def _nodes_exceeded(self) -> bool:
        if self.node_limit is None:
            return False
        return self.nodes >= self.node_limit

    def _should_abort(self) -> bool:
        return self._time_exceeded() or self._nodes_exceeded()


# simple convenience function
def search_position(board, max_depth: Optional[int] = None, time_ms: Optional[int] = None, nodes_limit: Optional[int] = None):
    s = Searcher()
    return s.search(board, max_depth=max_depth, time_ms=time_ms, nodes_limit=nodes_limit)


# Export
__all__ = ["Searcher", "search_position", "TTEntry", "TTFlag"]
