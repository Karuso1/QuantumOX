# QuantumOX

QuantumOX is a high-performance Tic-Tac-Toe engine written in **C++**, implementing the **UTTTI (Universal Tic-Tac-Toe Interface)** protocol. It draws inspiration from my earlier engine *QuantumKing*, and integrates advanced **Minimax** and **Negamax** search algorithms with **alpha-beta pruning**, enabling it to make strategic decisions efficiently—even on complex boards like 4x4, 5x5 or 3x3x3.

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
* Correctly identifies draws, wins, and losses
* Fully UTTTI-compliant (`go depth`, `grid emptygrid`, `setoption`, `stop`, etc.)
* Configurable board sizes: 3x3, 4x4, 3x3x3, etc.
* Lightweight, fast, and portable (pure C++)

## Compilation & Usage

To compile QuantumOX, use the included **Makefile** or compile manually with:

```bash
g++ -std=c++17 -O3 src/board.cpp src/engine.cpp src/search.cpp src/options.cpp src/main.cpp -o quantumox
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

option name Grid type string default 3x3
option name FirstPlayer type string default X
utttiok
setoption name Grid value 3x3
info string set "Grid" to 3x3
isready
readyok
utttinewgame
grid emptygrid
isready
readyok  
go depth 3
info depth 1 seldepth 1 score cp 36 nodes 10 minimaxpv 5 negamaxpv 5 pv 5
info depth 2 seldepth 2 score cp 12 nodes 37 minimaxpv 5 negamaxpv 5 pv 5 1      
info depth 3 seldepth 3 score cp 59 nodes 135 minimaxpv 5 negamaxpv 5 pv 5 1 3   
bestmove 5 ponder 1
```

OR (for playing first):

```
uttti
id name QuantumOX
id author Kartik

option name Grid type string default 3x3
option name FirstPlayer type string default X
utttiok
setoption name Grid value 3x3
info string set "Grid" to 3x3
isready
readyok
utttinewgame
grid emptygrid
grid emptygrid fill 2
isready
readyok
go depth 4
info depth 1 seldepth 1 score cp 17 nodes 9 minimaxpv 5 negamaxpv 5 pv 5
info depth 2 seldepth 2 score cp -28 nodes 36 minimaxpv 5 negamaxpv 5 pv 5 1     
info depth 3 seldepth 3 score cp 40 nodes 123 minimaxpv 5 negamaxpv 5 pv 5 7 9   
info depth 4 seldepth 4 score cp -30 nodes 330 minimaxpv 5 negamaxpv 5 pv 5 1 3 7
bestmove 5 ponder 1
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
