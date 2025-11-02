# Changelog

## 1.1 - 2025-10-31

- Introduced multi-threaded search for parallel move evaluation.
- Added a shared Transposition Table (TT) across threads for efficient position caching.
- Optimized evaluation and search flow, reducing average computation time per position.
- Improved alpha-beta pruning efficiency with enhanced move ordering.
- Minor internal improvements and build optimizations.

## 1.0 - 2025-10-22

- First release of the engine QuantumOX.
- Added full UTTTI (Universal Tic-Tac-Toe Interface) protocol support.
- Implemented hybrid Negamax with Alpha-beta search with iterative deepening.
- Introduced Zobrist hashing, killer moves, and history heuristics.
- Supports depth, node, and time-based search limits.
