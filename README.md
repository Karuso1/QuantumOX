# QuantumOX

**QuantumOX** is a highly advanced Tic-Tac-Toe engine using the **UTTTI(Universal Tic-Tac-Toe Interface)** protocol. Inspired by my previous engine **QuantumKing**, it combines a deep minimax/negamax search with alpha-beta pruning to make intelligent and efficient decisions—even in complex board configurations like 3x3, 4x4, or 3x3x3 grids.

QuantumOX analyzes positions thoroughly, predicts wins/losses/draws, and can handle manual or automated moves with precision. It’s designed to be engine-agnostic, letting other programs or interfaces communicate with it via standard UTTTI commands.

## Features

- Advanced Minimax/Negamax search with alpha-beta pruning
- Iterative deepening with depth control
- Principal Variation (PV) extraction for move insights
- Handles draws, wins, and losses accurately
- Fully UTTTI-compliant: supports go depth, grid emptygrid, setoption, stop, and more
- Configurable board sizes: 3x3, 4x4, 3x3x3, etc.
- Lightweight and portable (pure Python)

## Usage

Run the engine via Python:

```bash
python main.py
```

Basic commands:

- uttti - basic handshake
- setoption name ... value ... - edits the existing options
- isready - checks if the engine is ready
- utttinewgame - resets the tic tac toe board
- grid emptygrid fill ... - fills the O/X.
- go - starts the search, can add `depth {depth}` after it for customized depth.
- stop - stop current search
- quit / exit - exit engine
- help - show commands

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
uttinewgame
isready
readyok
grid emptygrid   
go depth 3
info depth 1 seldepth 1 score cp 36 nodes 10 pv 5
info depth 2 seldepth 2 score cp 12 nodes 37 pv 5 1      
info depth 3 seldepth 3 score cp 59 nodes 135 pv 5 1 3   
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
