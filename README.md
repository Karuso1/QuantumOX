# QuantumOX

QuantumOX is a high-performance Tic-Tac-Toe engine written in **C++**, implementing the **UTTTI (Universal Tic-Tac-Toe Interface)** protocol(inspired by UCI protocol). It draws inspiration from my earlier chess engine *QuantumKing*, and integrates advanced **Minimax** and **Negamax** search algorithms with **alpha-beta pruning**, enabling it to make strategic decisions efficiently—even on complex boards like 4x4, 5x5 or 3x3x3.

QuantumOX evaluates game states with precision, predicts outcomes (win/loss/draw), and supports both manual and automated play. Its modular design allows other programs or GUIs to interact with it using standard UTTTI commands.

> **NOTE:** Since there’s no dedicated GUI for UTTTI yet, you can manually play using: [Gametable](https://gametable.org/games/tic-tac-toe/) or [Math10](https://www.math10.com/en/math-games/tic-tac-toe/tic-tac-toe.html)

## Features

* Optimized Minimax/Negamax search with alpha-beta pruning
* Iterative deepening for adaptive search depth
* Principal Variation (PV) extraction for detailed move analysis
* Reports `minimaxpv`, `negamaxpv`, and combined `pv` in info lines:

  * `minimaxpv`: principal variation from the minimax perspective
  * `negamaxpv`: principal variation from the negamax perspective
  * `pv`: final chosen (by score) sequence after both analyses
  * `time`: total milliseconds elapsed since search start for that depth
* Correctly identifies draws, wins, and losses
* Fully UTTTI-compliant (`go depth`, `grid emptygrid`, `setoption`, `stop`, etc.)
* Configurable board sizes: 3x3, 4x4, 3x3x3, etc.
* Lightweight, fast, and portable (pure C++)

## Compilation & Usage

To compile QuantumOX, use the included **Makefile** or compile manually with:

```bash
g++ -std=c++17 -O3 src/*.cpp -o quantumox
```

Run the engine:

```bash
./quantumox
```

Basic commands:

* `uttti` - initialize the handshake
* `setoption name ... value ...` - edit existing options
* `isready` - check engine readiness
* `utttinewgame` - start a new game
* `grid emptygrid fill ...` - fill the board with moves
* `go depth {n}` - start searching up to depth `n`
* `stop` - stop current search
* `quit` / `exit` - exit engine
* `help` - show command list

Example UTTTI session:

```
uttti
id name QuantumOX
id author Kartik

option name Grid type combo default 3x3 var 3x3 var 4x4 var 5x5 var 6x6 var 7x7 var 8x8 var 15x15 var 3x3x3 var 4x4x4
option name FirstPlayer type combo default X var X var O
option name Hash type spin default 16 min 1 max 2097152
option name Threads type spin default 1 min 1 max 512
utttiok
isready
readyok
utttinewgame
grid emptygrid
grid emptygrid fill 1
isready
readyok
go depth 5
info string Using 1 thread
info depth 1 seldepth 2 score 12 nodes 18 nps 6000 hashfull 562 minimaxpv 5 negamaxpv 5 time 10 pv 5
info depth 2 seldepth 2 score 12 nodes 36 nps 3000 hashfull 562 minimaxpv 5 negamaxpv 5 time 18 pv 5
info depth 3 seldepth 12 score -39 nodes 336 nps 13440 hashfull 1000 minimaxpv 5 3 negamaxpv 5 3 time 32 pv 5 3
info depth 4 seldepth 2 score 69 nodes 363 nps 10676 hashfull 1000 minimaxpv 5 3 negamaxpv 6 5 time 41 pv 6 5
info depth 5 seldepth 12 score -39 nodes 847 nps 16607 hashfull 1000 minimaxpv 5 3 negamaxpv 5 7 time 56 pv 5 7
bestmove 5 ponder 7
grid emptygrid fill 1 5 7
isready
readyok
go depth 5
info string Using 1 thread
info depth 1 seldepth 2 score 31 nodes 14 nps 7000 hashfull 1000 minimaxpv 4 negamaxpv 4 time 10 pv 4
info depth 2 seldepth 2 score 31 nodes 28 nps 2545 hashfull 1000 minimaxpv 4 negamaxpv 4 time 16 pv 4
info depth 3 seldepth 4 score -20 nodes 102 nps 5666 hashfull 1000 minimaxpv 4 6 negamaxpv 4 6 time 24 pv 4 6
info depth 4 seldepth 2 score 90 nodes 123 nps 4730 hashfull 1000 minimaxpv 4 6 negamaxpv 6 4 time 31 pv 6 4
info depth 5 seldepth 9 score 20 nodes 266 nps 7600 hashfull 1000 minimaxpv 4 6 negamaxpv 4 6 8 time 41 pv 4 6 8
bestmove 4 ponder 6
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Karuso1/QuantumOX.git
cd QuantumOX
make -j build ARCH=x86-64 # only a place holder, you can see help when typing make
```

Run the compiled engine:

```bash
./quantumox
```

## About

QuantumOX is a personal project by **Kartik**. It’s a demonstration that even simple games like Tic-Tac-Toe can showcase the depth and efficiency of well-designed AI search algorithms.

## License

QuantumOX is licensed under the **GPL-3.0 License.**
You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, as long as you include the original copyright notice.

For the full license text, see [LICENSE](https://github.com/Karuso1/QuantumOX/blob/main/LICENSE).
