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

#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <vector>
#include <optional>
#include <utility>

#include "board.h"
#include "search.h"
#include "options.h"
#include "utils.h"

namespace QuantumOX {

    // Struct returned by go()
    struct GoResult {
        std::vector<std::string> info_lines; // UTTTI info lines
        std::string bestmove_line;           // final bestmove string
        std::optional<int> bestmove;         // chosen bestmove
        Searcher::SearchResult raw;          // raw search result
    };
    
    class QuantumOXEngine {
    public:
        QuantumOXEngine();
    
        // Options / game lifecycle
        std::pair<bool, std::string> set_option(const std::string& name, const std::string& value);
        void new_game();
        std::string play_moves(const std::vector<int>& moves);
        std::string play_move(int move);
    
        // Search / go
        GoResult go(std::optional<int> depth = std::nullopt,
                    std::optional<int> time_ms = std::nullopt,
                    std::optional<int> nodes = std::nullopt);
        
        void stop();
        
        // Utilities
        std::string board_state() const;
        std::vector<int> legal_moves() const;
        
    private:
        Board board;
        Searcher searcher;
    };

} // namespace QuantumOX

#endif // ENGINE_H
