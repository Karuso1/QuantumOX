# QuantumOX

QuantumOX is a highly advanced Tic-Tac-Toe engine using the UTTTI (Universal Tic-Tac-Toe Interface) protocol. Inspired by my previous engine QuantumKing, it combines a deep minimax/negamax search with alpha-beta pruning to make intelligent and efficient decisions—even in complex board configurations like 3x3, 4x4, or 3x3x3 grids.

QuantumOX analyzes positions thoroughly, predicts wins/losses/draws, and can handle manual or automated moves with precision. It’s designed to be engine-agnostic, letting other programs or interfaces communicate with it via standard UTTTI commands.

> NOTE: Since there is no Tic-Tac-Toe GUI for UTTTI, play manually in: [Gametable](https://gametable.org/games/tic-tac-toe/) or [Math10](https://www.math10.com/en/math-games/tic-tac-toe/tic-tac-toe.html)

## Features

- Advanced Minimax/Negamax search with alpha-beta pruning
- Iterative deepening with depth control
- Principal Variation (PV) extraction for move insights
- Reports `minimaxpv` and `negamaxpv` in info lines for detailed search analysis(new feature):
  - `minimaxpv` shows the principal variation according to the minimax evaluation
  - `negamaxpv` shows the principal variation according to the negamax evaluation
  - `pv` now shows the final selected move sequence after combining insights from both
- Handles draws, wins, and losses accurately
- Fully UTTTI-compliant: supports `go depth`, `grid emptygrid`, `setoption`, `stop`, and more
- Configurable board sizes: 3x3, 4x4, 3x3x3, etc.
- Lightweight and portable (pure Python)

## Usage

Run the engine via Python:

```bash
python main.py
```

Basic commands:

- `uttti` - basic handshake
- `setoption name ... value ...` - edits the existing options
- `isready` - checks if the engine is ready
- `utttinewgame` - resets the tic tac toe board
- `grid emptygrid fill ...` - fills the O/X.
- `go` - starts the search, can add `depth {depth}` after it for customized depth.
- `stop` - stop current search
- `quit` / exit - exit engine
- `help` - shows commands

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

QuantumOX is pure Python - no extra dependencies are needed. Clone the repo and run:

```
git clone https://github.com/Karuso1/QuantumOX.git
cd QuantumOX
python main.py
```

## About

QuantumOX is a personal project by **Kartik**. It's a proof-of-concept that even simple games like Tic-Tac-Toe can benefit from serious AI techniques.

## License

QuantumOX is released under the **MIT License.**
You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, as long as you include the original copyright notice.

For full license text, see [LICENSE](https://github.com/Karuso1/QuantumOX/blob/main/LICENSE)
