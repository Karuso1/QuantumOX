/* 
 * QuantumOX, a Tic Tac Toe engine supporting UTTTI.
 * Copyright (C) 2025 Kartik Pawar
 *
 * QuantumOX is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * QuantumOX is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>
#include <array>
#include <vector>
#include <sstream>
#include <stdexcept>

namespace QuantumOX {

    // Engine identity
    inline constexpr auto ENGINE_NAME = "QuantumOX";
    inline constexpr auto ENGINE_VERSION = "1.1";
    inline constexpr auto ENGINE_AUTHOR = "Kartik";
    inline constexpr auto UTTTI_VERSION = "0.1";
    
    // Default options
    inline constexpr auto DEFAULT_GRID = "3x3";
    inline constexpr auto DEFAULT_THREADS = 1; // configure the default threads if you want
    inline constexpr auto DEFAULT_HASH = 16;
    inline const std::array<std::string, 9> SUPPORTED_GRIDS = {"3x3", "4x4", "5x5", "6x6", "7x7", "8x8", "15x15", "3x3x3", "4x4x4"};
    
    // Player/square symbols
    inline constexpr auto SYMBOL_EMPTY = '.';
    inline constexpr auto SYMBOL_X = 'X';
    inline constexpr auto SYMBOL_O = 'O';
    
    // Search score sentinels
    inline constexpr int SCORE_MATE = 10'000'000;
    inline constexpr int SCORE_WIN = 1'000'000;
    inline constexpr int SCORE_DRAW = 0;
    inline constexpr int SCORE_LOSS = -SCORE_WIN;
    
    // UTTTI protocol keywords / tokens
    inline constexpr auto CMD_UTTTI = "uttti";
    inline constexpr auto CMD_ID = "id";
    inline constexpr auto CMD_ISREADY = "isready";
    inline constexpr auto CMD_READYOK = "readyok";
    inline constexpr auto CMD_SETOPTION = "setoption";
    inline constexpr auto CMD_INFO = "info";
    inline constexpr auto CMD_NEWGAME = "utttinewgame";
    inline constexpr auto CMD_GRID = "grid";
    inline constexpr auto CMD_GO = "go";
    inline constexpr auto CMD_BESTMOVE = "bestmove";
    
    // Info keys the engine will emit
    inline const std::array<std::string, 5> INFO_KEYS = {"depth", "seldepth", "score", "nodes", "pv"};
    
    // Search limit keywords accepted by `go` command
    inline const std::array<std::string, 4> SEARCH_LIMITS = {"depth", "movetime", "nodes", "infinite"};
    
    // Default search settings
    inline constexpr int DEFAULT_MAX_DEPTH = 80;
    inline constexpr int DEFAULT_TIME_MS = 1000;
    
    // Misc / formatting
    inline constexpr auto PV_SEPARATOR = " ";
    inline constexpr auto INFO_STRING_PREFIX = "info string";
    
    // --- tiny helper: parse grid spec -------------------------------------------
    inline std::vector<int> parse_grid_spec(const std::string& spec) {
        std::vector<int> result;
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, 'x')) {
            try {
                result.push_back(std::stoi(item));
            } catch (const std::invalid_argument&) {
                throw std::runtime_error("Invalid grid spec: " + spec + ". Expected format like '3x3' or '3x3x3'.");
            }
        }
        return result;
    }
    
    // Backwards-compatible alias
    inline std::vector<int> grid_dims(const std::vector<int>& dims) {
        return dims;
    }
    
    inline std::vector<int> grid_dims(const std::string& spec) {
        return parse_grid_spec(spec);
    }

} // namespace QuantumOX

#endif // CONSTANTS_H
