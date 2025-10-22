# tests/test_search.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search import Searcher, search_position

# ------------------- Mock Board for testing -------------------
class MockBoard:
    """
    Minimal board compatible with search.py for testing.
    3x3 grid (cells 0-8), side_to_move 'X' or 'O'.
    """
    def __init__(self):
        self.cells = [0] * 9  # 0=empty, 1=X, 2=O
        self.side_to_move = 'X'
        self.moves_made = []

    def legal_moves(self):
        return [i+1 for i, c in enumerate(self.cells) if c == 0]

    def make_move(self, move: int):
        self.cells[move-1] = 1 if self.side_to_move=='X' else 2
        self.moves_made.append(move)
        self.side_to_move = 'O' if self.side_to_move=='X' else 'X'

    def unmake_move(self, move: int):
        self.cells[move-1] = 0
        self.moves_made.pop()
        self.side_to_move = 'O' if self.side_to_move=='X' else 'X'

    def is_win(self, player: str):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        target = 1 if player=='X' else 2
        return any(all(self.cells[i]==target for i in line) for line in wins)

    def is_draw(self):
        return all(c != 0 for c in self.cells) and not self.is_win('X') and not self.is_win('O')

    def evaluate(self, player: str):
        # simple scoring: +10 for X win, -10 for O win, else 0
        if self.is_win('X'):
            return 10 if player=='X' else -10
        if self.is_win('O'):
            return 10 if player=='O' else -10
        return 0

    def zobrist_key(self):
        return tuple(self.cells) + (self.side_to_move,)

# ------------------- Pytest-style test -------------------
def test_search_position_runs():
    board = MockBoard()
    result = search_position(board, max_depth=2, nodes_limit=50)
    assert "bestmove" in result
    assert result["bestmove"] is not None, "Searcher did not find a move!"
    assert "score" in result
    assert "pv" in result
    assert "nodes" in result
    print("Test search result:", result)

# ------------------- Optional main for direct run -------------------
if __name__ == "__main__":
    board = MockBoard()
    print("Running search_position on MockBoard...")
    result = search_position(board, max_depth=2, nodes_limit=50)
    print("Result:", result)
