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

    def _quick_move_score(self, board, mv: int, for_player: str) -> int:
        """Make move, call board.evaluate(for_player) if available, then unmake.
        Returns 0 on any exception — cheap ordering heuristic.
        """
        try:
            board.make_move(mv)
            try:
                val = board.evaluate(for_player)
            except Exception:
                val = 0
        finally:
            try:
                board.unmake_move(mv)
            except Exception:
                # if unmake fails, we can't recover — but preserve original behavior
                pass
        return val

    # ----------------- public API -------------------------------------------
    def search(
        self,
        board,
        max_depth: Optional[int] = None,
        time_ms: Optional[int] = None,
        nodes_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run iterative deepening and compare negamax vs minimax each depth,
        picking the best-scoring move between them.

        Returns:
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
        best_pv: List[int] = []
        infos: List[Dict[str, Any]] = []

        key_plain = self._key(board)
        root_player = board.side_to_move

        # aspiration window params (centipawns)
        ASP_WINDOW = 50
        prev_neg_score = 0
        prev_min_score = 0

        for depth in range(1, max_depth + 1):
            if self._should_abort():
                break

            # ----- NEGAMAX pass with aspiration window -----
            nodes_before = self.nodes
            # if we have a previous score, use a small window
            if depth > 1:
                alpha = prev_neg_score - ASP_WINDOW
                beta = prev_neg_score + ASP_WINDOW
            else:
                alpha = -10_000_000
                beta = 10_000_000

            neg_score = self._negamax_root(board, depth, alpha, beta)

            # if aspiration failed (score outside window), re-search with full window
            if depth > 1 and (neg_score <= alpha or neg_score >= beta):
                alpha = -10_000_000
                beta = 10_000_000
                neg_score = self._negamax_root(board, depth, alpha, beta)

            neg_nodes = self.nodes - nodes_before
            prev_neg_score = neg_score

            if self.abort:
                break

            # get negamax best move & pv
            neg_move = None
            neg_pv: List[int] = []
            try:
                ent = self.tt.get(self._key(board))
                neg_move = ent.best_move if ent else None
            except Exception:
                neg_move = None
            # try to build PV (fall back gracefully)
            try:
                neg_pv = self._build_pv(board)
            except Exception:
                neg_pv = [neg_move] if neg_move else []

            if self._should_abort():
                break

            # ----- MINIMAX pass with aspiration window -----
            nodes_before_min = self.nodes
            if depth > 1:
                alpha = prev_min_score - ASP_WINDOW
                beta = prev_min_score + ASP_WINDOW
            else:
                alpha = -10_000_000
                beta = 10_000_000

            min_score = self._minimax_root(board, depth, alpha, beta)
            if depth > 1 and (min_score <= alpha or min_score >= beta):
                alpha = -10_000_000
                beta = 10_000_000
                min_score = self._minimax_root(board, depth, alpha, beta)

            min_nodes = self.nodes - nodes_before_min
            prev_min_score = min_score

            if self.abort:
                break

            # get minimax best move & pv
            min_move = None
            min_pv: List[int] = []
            try:
                min_ent = self.tt.get((self._key(board), root_player))
                min_move = min_ent.best_move if min_ent else None
            except Exception:
                min_move = None
            # try to build PV for minimax (fall back to plain pv builder)
            try:
                min_pv = self._build_pv_for_root(board, root_player)
            except Exception:
                try:
                    min_pv = self._build_pv(board)
                except Exception:
                    min_pv = [min_move] if min_move else []

            # ----- pick the best move between negamax and minimax -----
            # Both scores are from the perspective of the root side-to-move; compare directly.
            selector = "negamax"
            chosen_score = neg_score
            chosen_move = neg_move
            chosen_pv = neg_pv
            chosen_nodes = neg_nodes

            if min_score > neg_score:
                selector = "minimax"
                chosen_score = min_score
                chosen_move = min_move
                chosen_pv = min_pv
                chosen_nodes = min_nodes
            elif min_score == neg_score:
                # tie-breaker: prefer the one with fewer nodes, then prefer negamax
                if min_nodes < neg_nodes:
                    selector = "minimax"
                    chosen_score = min_score
                    chosen_move = min_move
                    chosen_pv = min_pv
                    chosen_nodes = min_nodes

            # update best so-far
            if chosen_move is not None:
                best_move = chosen_move
                best_score = chosen_score
                best_pv = chosen_pv

            infos.append({
                "depth": depth,
                "seldepth": depth,
                "score": best_score,
                "nodes": self.nodes,
                "negamaxpv": neg_pv,
                "minimaxpv": min_pv,
                "pv": best_pv,
            })

            # time/node limit check
            if self._time_exceeded() or self._nodes_exceeded():
                break

        return {
            "bestmove": best_move,
            "score": best_score,
            "pv": best_pv,
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

    def _minimax_root(self, board, depth: int, alpha: int, beta: int) -> int:
        """
        Root wrapper for a minimax (max/min) implementation with alpha-beta.
        Returns score FROM THE PERSPECTIVE OF root_player (the side to move at root).
        """
        self.nodes += 1
        root_player = board.side_to_move
        key_plain = self._key(board)
        tt_key = (key_plain, root_player)

        # init killer/history structures on demand
        if not hasattr(self, "killer_moves"):
            self.killer_moves = {}  # depth -> list[moves]
        if not hasattr(self, "history"):
            self.history = {}  # move -> score

        moves = board.legal_moves()
        if not moves:
            return self._evaluate_for_root(board, root_player)

        # TT move ordering
        tt_entry = self.tt.get(tt_key)
        tt_move = tt_entry.best_move if tt_entry else None
        if tt_move and tt_move in moves:
            moves.remove(tt_move)
            moves = [tt_move] + moves

        # Pre-check immediate winning moves and sort using heuristics
        winning = []
        others = []
        for mv in moves:
            board.make_move(mv)
            # The player who just moved is the opponent of current side_to_move
            prev_player = 'O' if board.side_to_move == 'X' else 'X'
            is_win = board.is_win(prev_player)
            board.unmake_move(mv)
            if is_win:
                winning.append(mv)
            else:
                others.append(mv)

        # If a winning move exists, prefer it immediately
        ordered = winning + others

        # Further ordering using killer and history heuristics
        def move_score(mv):
            score = 0
            # TT move high priority
            if mv == tt_move:
                score += 10_000_000
            # killer moves
            for d, kms in (self.killer_moves.items() if hasattr(self, "killer_moves") else []):
                if mv in kms:
                    score += 1000
            # history heuristic
            score += self.history.get(mv, 0)
            # small tie-breaker by move index
            score += -mv * 0.01
            return score

        ordered = sorted(ordered, key=move_score, reverse=True)

        best_score = -10_000_000
        best_move = None
        original_alpha = alpha

        for mv in ordered:
            if self._should_abort():
                break
            board.make_move(mv)
            score = self._minimax(board, depth - 1, alpha, beta, root_player)
            board.unmake_move(mv)

            if score > best_score:
                best_score = score
                best_move = mv

            alpha = max(alpha, score)
            if alpha >= beta:
                # beta cutoff: record killer and history heuristics
                kms = self.killer_moves.setdefault(depth, [])
                if mv not in kms:
                    kms.append(mv)
                    if len(kms) > 2:
                        kms.pop(0)
                self.history[mv] = self.history.get(mv, 0) + (1 << depth)
                # store as LOWER bound
                self._store_tt(tt_key, depth, best_score, TTFlag.LOWER, best_move)
                break

        # determine flag
        if best_score <= original_alpha:
            flag = TTFlag.UPPER
        elif best_score >= beta:
            flag = TTFlag.LOWER
        else:
            flag = TTFlag.EXACT

        self._store_tt(tt_key, depth, best_score, flag, best_move)
        # Also store a plain-key entry so existing PV builder (that uses self._key(board))
        # can still reconstruct PVs if desired (best_move only).
        try:
            self._store_tt(key_plain, depth, best_score, flag, best_move)
        except Exception:
            pass

        return best_score
    
    def _minimax(self, board, depth: int, alpha: int, beta: int, root_player: str) -> int:
        """
        Minimax with alpha-beta pruning.
        - `root_player` is the original side to move at root; scores are returned
          from the perspective of root_player (higher is better for root_player).
        - At each node we treat it as maximizing if board.side_to_move == root_player,
          otherwise minimizing.
        """
        self.nodes += 1
        if self._should_abort():
            self.abort = True
            return 0

        key_plain = self._key(board)
        tt_key = (key_plain, root_player)

        # TT probe
        tt_entry = self.tt.get(tt_key)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TTFlag.EXACT:
                return tt_entry.score
            elif tt_entry.flag == TTFlag.LOWER and tt_entry.score >= beta:
                return tt_entry.score
            elif tt_entry.flag == TTFlag.UPPER and tt_entry.score <= alpha:
                return tt_entry.score

        # terminal or depth==0: evaluate from root_player perspective
        if depth == 0 or board.is_win('X') or board.is_win('O') or board.is_draw():
            val = self._evaluate_for_root(board, root_player)
            self._store_tt(tt_key, depth, val, TTFlag.EXACT, None)
            return val

        # generate moves
        moves = board.legal_moves()
        if not moves:
            val = self._evaluate_for_root(board, root_player)
            self._store_tt(tt_key, depth, val, TTFlag.EXACT, None)
            return val

        # lazy init for heuristics
        if not hasattr(self, "killer_moves"):
            self.killer_moves = {}
        if not hasattr(self, "history"):
            self.history = {}

        # TT move ordering
        tt_plain_entry = self.tt.get(tt_key)
        tt_move = tt_plain_entry.best_move if tt_plain_entry else None
        if tt_move and tt_move in moves:
            moves.remove(tt_move)
            moves = [tt_move] + moves

        # prefer immediate winning moves first (fast check)
        winning = []
        others = []
        for mv in moves:
            board.make_move(mv)
            prev_player = 'O' if board.side_to_move == 'X' else 'X'
            is_win = board.is_win(prev_player)
            board.unmake_move(mv)
            if is_win:
                winning.append(mv)
            else:
                others.append(mv)

        ordered = winning + others

        # ordering by heuristics: TT, killer, history
        def mv_sort_key(mv):
            s = 0
            if mv == tt_move:
                s += 10_000_000
            if mv in self.killer_moves.get(depth, []):
                s += 1000
            s += self.history.get(mv, 0)
            s += -mv * 0.01
            return s

        ordered = sorted(ordered, key=mv_sort_key, reverse=True)

        maximizing = (board.side_to_move == root_player)

        best_score = -10_000_000 if maximizing else 10_000_000
        best_move = None
        original_alpha = alpha
        original_beta = beta

        for mv in ordered:
            if self._should_abort():
                self.abort = True
                break
            board.make_move(mv)
            score = self._minimax(board, depth - 1, alpha, beta, root_player)
            board.unmake_move(mv)

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = mv
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = mv
                beta = min(beta, score)

            # cutoff
            if alpha >= beta:
                # killer + history
                kms = self.killer_moves.setdefault(depth, [])
                if mv not in kms:
                    kms.append(mv)
                    if len(kms) > 2:
                        kms.pop(0)
                self.history[mv] = self.history.get(mv, 0) + (1 << depth)

                # store LOWER bound if maximizing cut, UPPER if minimizing cut (consistent with score semantics)
                if maximizing:
                    store_flag = TTFlag.LOWER
                else:
                    store_flag = TTFlag.UPPER
                self._store_tt(tt_key, depth, best_score, store_flag, best_move)
                break

        # set final flag
        if maximizing:
            if best_score <= original_alpha:
                flag = TTFlag.UPPER
            elif best_score >= original_beta:
                flag = TTFlag.LOWER
            else:
                flag = TTFlag.EXACT
        else:
            # for minimizing nodes the inequalities flip
            if best_score >= original_beta:
                flag = TTFlag.LOWER
            elif best_score <= original_alpha:
                flag = TTFlag.UPPER
            else:
                flag = TTFlag.EXACT

        self._store_tt(tt_key, depth, best_score, flag, best_move)
        # also mirror best_move into plain key for PV reconstruction (non-invasive)
        try:
            self._store_tt(key_plain, depth, best_score, flag, best_move)
        except Exception:
            pass

        return best_score


    # helper: building PV when searching with a specific root_player
    def _build_pv_for_root(self, board, root_player: str) -> List[int]:
        """Reconstruct PV using TT entries stored with (key, root_player).
        Uses make/unmake on the original board (no deepcopy) for performance.
        Restores board state before returning.
        """
        pv: List[int] = []
        played: List[int] = []
        try:
            while True:
                k = (self._key(board), root_player)
                ent = self.tt.get(k)
                if not ent or not ent.best_move:
                    break
                mv = ent.best_move
                # safety: if move invalid, break
                legal = board.legal_moves()
                if mv not in legal:
                    break
                pv.append(mv)
                board.make_move(mv)
                played.append(mv)
                # guard: avoid infinite loops
                if len(pv) > 256:
                    break
        finally:
            # unmake everything we played to restore original board
            for mv in reversed(played):
                try:
                    board.unmake_move(mv)
                except Exception:
                    # if unmake fails, give up restoring but avoid crashing
                    pass
        return pv
    
    # helper: evaluate from root_player perspective (uses board.evaluate if present)
    def _evaluate_for_root(self, board, root_player: str) -> int:
        try:
            return board.evaluate(root_player)
        except Exception:
            # terminal-only fallback
            if board.is_win(root_player):
                return SCORE_WIN
            opp = 'O' if root_player == 'X' else 'X'
            if board.is_win(opp):
                return SCORE_LOSS
            if board.is_draw():
                return SCORE_DRAW
            return 0

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
