# QuantumOX

QuantumOX is a high-performance Tic-Tac-Toe engine written in **C++**, implementing the **UTTTI (Universal Tic-Tac-Toe Interface)** protocol (inspired by UCI protocol). It draws inspiration from my earlier chess engine *QuantumKing*, and integrates advanced search algorithms with **alpha-beta pruning**, enabling it to make strategic decisions efficiently—even on complex boards like 4x4, 5x5 or 3x3x3.

> **NOTE:** Since there’s no dedicated GUI for UTTTI yet, you can manually play using: [Gametable](https://gametable.org/games/tic-tac-toe/) or [Math10](https://www.math10.com/en/math-games/tic-tac-toe/tic-tac-toe.html)

## Features

* Optimized search with alpha-beta pruning
* Iterative deepening for adaptive search depth
* Principal Variation (PV) extraction for detailed move analysis
* Reports combined `pv` in info lines:

  * `pv`: final chosen sequence by score
  * `score`: evaluation of the current board
  * `depth` / `seldepth`: search progress info
  * `nodes` / `nps` / `time` for performance stats
* Correctly identifies draws, wins, and losses
* Fully UTTTI-compliant (`go depth`, `grid emptygrid`, `setoption`, `stop`, etc.)
* Configurable board sizes: 3x3, 4x4, 3x3x3, etc.
* Lightweight, fast, and portable (pure C++)

> **Note:** Algorithm switching between different internal search strategies is entirely internal; only the final chosen PV and bestmove are exposed.

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
go depth 5
info string Using 1 thread
info depth 1 seldepth 2 score 12 nodes 18 nps 6000 hashfull 562 time 8 pv 5
info depth 2 seldepth 2 score 12 nodes 36 nps 3600 hashfull 562 time 14 pv 5
info depth 3 seldepth 12 score -39 nodes 336 nps 16000 hashfull 1000 time 26 pv 5 3
info depth 4 seldepth 2 score 69 nodes 363 nps 12964 hashfull 1000 time 33 pv 6 5
info depth 5 seldepth 12 score -39 nodes 847 nps 19250 hashfull 1000 time 48 pv 5 7
bestmove 5 ponder 7
grid emptygrid fill 1 5 7
go depth 5
info string Using 1 thread
info depth 1 seldepth 2 score 31 nodes 14 nps 7000 hashfull 1000 time 8 pv 4
info depth 2 seldepth 2 score 31 nodes 28 nps 3111 hashfull 1000 time 14 pv 4
info depth 3 seldepth 4 score -20 nodes 102 nps 6375 hashfull 1000 time 21 pv 4 6
info depth 4 seldepth 2 score 90 nodes 123 nps 5347 hashfull 1000 time 27 pv 6 4
info depth 5 seldepth 9 score 20 nodes 266 nps 8580 hashfull 1000 time 36 pv 4 6 8
bestmove 4 ponder 6
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Karuso1/QuantumOX.git
cd QuantumOX
make -j build ARCH=x86-64 # placeholder, see `make` for options
```

Run the compiled engine:

```bash
./quantumox
```

## About

QuantumOX is a personal project by **Kartik**. It demonstrates that even simple games like Tic-Tac-Toe can showcase the depth and efficiency of well-designed AI search algorithms.

## License

QuantumOX is licensed under the **GPL-3.0 License**. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, as long as you include the original copyright notice.

For the full license text, see [LICENSE](https://github.com/Karuso1/QuantumOX/blob/main/LICENSE).
