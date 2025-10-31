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
  * `pv`: final hybrid sequence after both analyses
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

option name Grid type combo default 3x3 var 3x3 var 4x4 var 5x5 var 3x3x3
option name FirstPlayer type combo default X var X var O
utttiok
isready
readyok
utttinewgame
grid emptygrid
grid emptygrid fill 1
isready
readyok
go depth 5
info depth 1 seldepth 1 score 12 nodes 18 minimaxpv 5 negamaxpv 5 time 0 pv 5
info depth 2 seldepth 2 score -39 nodes 93 minimaxpv 5 3 negamaxpv 5 3 time 0 pv 5 3
info depth 3 seldepth 3 score 31 nodes 286 minimaxpv 5 3 2 negamaxpv 5 3 2 time 1 pv 5 3 2
info depth 4 seldepth 4 score -40 nodes 741 minimaxpv 5 9 4 3 negamaxpv 5 9 2 7 time 3 pv 5 9 2 7
info depth 5 seldepth 5 score 20 nodes 1645 minimaxpv 5 3 2 8 4 negamaxpv 5 3 2 8 4 time 5 pv 5 3 2 8 4
bestmove 5 ponder 3
grid emptygrid fill 1 5 3
isready
readyok
go depth 5
info depth 1 seldepth 1 score 20 nodes 14 minimaxpv 2 8 4 negamaxpv 8 time 0 pv 2 8 4
info depth 2 seldepth 2 score 20 nodes 33 minimaxpv 2 8 4 negamaxpv 2 8 4 time 0 pv 2 8 4
info depth 3 seldepth 3 score 20 nodes 52 minimaxpv 2 8 4 negamaxpv 2 8 4 time 0 pv 2 8 4
info depth 4 seldepth 4 score -40 nodes 172 minimaxpv 2 8 7 6 negamaxpv 2 8 7 6 time 0 pv 2 8 7 6
info depth 5 seldepth 5 score 0 nodes 339 minimaxpv 2 8 7 6 9 negamaxpv 2 8 7 6 9 time 0 pv 2 8 7 6 9
bestmove 2 ponder 8
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
