# test.py

from engine import QuantumOXEngine

def test_engine_initialization():
    """Check that the engine initializes without errors."""
    engine = QuantumOXEngine()
    assert engine.board is not None

def test_new_game_resets_board():
    """Check that starting a new game resets the board."""
    engine = QuantumOXEngine()
    engine.new_game()
    # You can check if board is empty; adapt depending on your Board API
    assert len(engine.board.legal_moves()) > 0

def test_dummy():
    """A simple dummy test to ensure pytest runs."""
    assert True
