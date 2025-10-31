# Universal Tic-Tac-Toe Interface (UTTTI) Specification

## Overview

UTTTI (Universal Tic-Tac-Toe Interface) is a communication protocol between Tic-Tac-Toe engines and graphical or command-line interfaces. It is inspired by UCI (Universal Chess Interface) and defines a structured way for engines like **QuantumOX** to exchange information with user interfaces.

QuantumOX is the **first engine** to implement and demonstrate the UTTTI protocol.

## Initialization

When the engine starts, it should identify itself and list available configuration options:

```
uttti
id name QuantumOX
id author Kartik

option name Grid type combo default 3x3 var 3x3 var 4x4 var 5x5 var 3x3x3
option name FirstPlayer type combo default X var X var O
option name UseTranspositionTable type check default true
option name SearchAlgorithm type combo default Negamax var Negamax var Minimax
option name MaxDepth type spin default 5 min 1 max 10
option name UseOXNN type check default false
utttiok
```

### Notes

* The **options** are user-configurable settings that can vary between engines.
* The **OXNN** option enables a neural evaluator module for board scoring.
* Blank lines between sections are optional.

## Basic Commands

### `isready`

Used by the interface to check if the engine is ready to receive new commands.

**Example:**

```
isready
readyok
```

### `utttinewgame`

Signals the start of a new game. The engine should reset all internal states (hash tables, history, etc.).

**Example:**

```
utttinewgame
```

### `grid`

Defines the board state.

#### `grid emptygrid`

Starts from an empty grid.

#### `grid emptygrid fill <moves>`

Populates the grid with a list of moves. The order determines player alternation.

**If the player moves first:**

```
grid emptygrid fill <player's move> <engine's move> <player's move> ...
```

**If the engine plays first after `grid emptygrid`:**

```
grid emptygrid fill <engine's best move> <player's move> <engine's move> ...
```

## Search Commands

### `go`

Starts the search. The interface can specify limits such as depth, nodes, or time.

**Example:**

```
go depth 5
```

### Search Output

The engine may optionally print **info lines** during the search:

```
info depth 1 seldepth 1 score 12 nodes 18 minimaxpv 5 negamaxpv 5 time 0 pv 5
info depth 2 seldepth 2 score -39 nodes 93 minimaxpv 5 3 negamaxpv 5 3 time 0 pv 5 3
```

### Best Move

At the end of the search, the engine outputs:

```
bestmove 5 ponder 3
```

or simply:

```
bestmove 3
```

* `bestmove` indicates the move chosen by the engine.
* `ponder` is optional and can specify the move expected from the opponent.

## Extensibility

* Implementers can define additional custom commands and options as needed.
* `info` lines are optional but recommended for debug or GUI integration.
* Future extensions may include neural evaluation modules (like **OXNN**) or multi-threading support.

**Version:** 0.1
**Maintained by:** Kartik
**Used in:** QuantumOX Engine
