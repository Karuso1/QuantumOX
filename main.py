"""
main.py

UTTTI main loop for QuantumOX. Reads commands from stdin and speaks UTTTI on stdout.

Author: Kartik (QuantumOX project)
"""
from __future__ import annotations

import sys
from typing import List

from constants import ENGINE_NAME, ENGINE_AUTHOR
from utils import tokenize_command, parse_setoption, parse_move_token
from options import list_options
from engine import QuantumOXEngine

HELP_TEXT = """
QuantumOX UTTTI commands:
  uttti                     - handshake
  setoption name <name> value <value>  - set engine option
  isready                   - check if engine is ready
  utttinewgame              - reset board silently
  grid emptygrid [fill ...] - reset board and optionally fill moves
  go depth <N>              - run search to depth N
  quit / exit               - exit engine
  help                      - show this help text
"""

def handle_uttti(engine: QuantumOXEngine) -> None:
    print(f"id name {ENGINE_NAME}")
    print(f"id author {ENGINE_AUTHOR}")
    print("")  # blank line between id and options
    opts = list_options()
    for name, meta in opts.items():
        line = f"option name {name} type {meta['type']} default {meta['default']}"
        if meta['type'] == 'spin':
            if meta.get('min') is not None:
                line += f" min {meta['min']}"
            if meta.get('max') is not None:
                line += f" max {meta['max']}"
        print(line)
    print("utttiok")
    sys.stdout.flush()

def handle_setoption(engine: QuantumOXEngine, tokens: List[str]):
    name, value = parse_setoption(tokens)
    if name is None:
        return
    success, msg = engine.set_option(name, value)
    if success:
        print(f'info string {msg}')
    else:
        print(msg)
    sys.stdout.flush()

def handle_grid(engine: QuantumOXEngine, tokens: List[str]):
    if len(tokens) < 2:
        return
    sub = tokens[1].lower()
    if sub == "emptygrid":
        engine.new_game()  # silent reset
        if len(tokens) >= 4 and tokens[2].lower() == "fill":
            raw_moves = tokens[3:]
            moves = []
            for t in raw_moves:
                try:
                    moves.append(parse_move_token(t))
                except Exception:
                    return
            engine.play_moves(moves)  # silent, no info string


def handle_go(engine: QuantumOXEngine, tokens: List[str]):
    depth = None
    movetime = None
    nodes_limit = None
    i = 1
    while i < len(tokens):
        t = tokens[i].lower()
        if t == "depth" and i + 1 < len(tokens):
            try: depth = int(tokens[i + 1])
            except ValueError: depth = None
            i += 2
            continue
        if t in ("movetime", "movetime_ms") and i + 1 < len(tokens):
            try: movetime = int(tokens[i + 1])
            except ValueError: movetime = None
            i += 2
            continue
        if t == "nodes" and i + 1 < len(tokens):
            try: nodes_limit = int(tokens[i + 1])
            except ValueError: nodes_limit = None
            i += 2
            continue
        i += 1

    # Check if there are any legal moves
    legal = engine.board.legal_moves()
    if not legal:
        print("bestmove 0000")  # no moves available
        sys.stdout.flush()
        return

    res = engine.go(depth=depth, time_ms=movetime, nodes=nodes_limit)
    for line in res.get("info_lines", []):
        print(line)
        sys.stdout.flush()
    print(res.get("bestmove_line", "bestmove 0"))
    sys.stdout.flush()

def main():
    engine = QuantumOXEngine()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = tokenize_command(line)
        if not tokens:
            continue
        cmd = tokens[0].lower()
        try:
            if cmd == "uttti":
                handle_uttti(engine)
            elif cmd == "setoption":
                handle_setoption(engine, tokens)
            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()
            elif cmd == "utttinewgame":
                engine.new_game()  # silent reset
            elif cmd == "grid":
                handle_grid(engine, tokens)  # silent fill
            elif cmd == "go":
                handle_go(engine, tokens)
            elif cmd == "stop":
                engine.stop() # stop ongoing search
            elif cmd == "help":
                print(HELP_TEXT.strip())
                sys.stdout.flush()
            elif cmd in ("quit", "exit"):
                break
            else:
                print(f"Unknown command: {tokens[0]}, type 'help' for UTTTI commands")
                sys.stdout.flush()
        except Exception:
            continue


if __name__ == "__main__":
    main()
