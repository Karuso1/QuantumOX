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

#include "engine.h"

#include <string>
#include <vector>
#include <optional>
#include <sstream>
#include <iostream>

namespace QuantumOX {

QuantumOXEngine::QuantumOXEngine() {
    std::string grid;
    try {
        grid = get_option("Grid");
    } catch (...) {
        grid = DEFAULT_GRID;
    }
    board = Board(grid);

    // Attempt to set starting player from options if available
    try {
        std::string fp = get_option("FirstPlayer");
        // board.set_side_to_move(fp); // uncomment if Board exposes setter
    } catch (...) {
        // ignore
    }
}

// ------------------ options / game lifecycle ---------------------------

std::pair<bool,std::string> QuantumOXEngine::set_option(const std::string& name, const std::string& value) {
    auto res = ::QuantumOX::set_option(name, value);
    bool success = res.first;
    std::string msg = res.second;

    if (success && name == "Grid") {
        board = Board(value);
        try {
            std::string fp = get_option("FirstPlayer");
            // board.set_side_to_move(fp);
        } catch(...) {}
    }

    return {success, msg};
}

void QuantumOXEngine::new_game() {
    std::string grid = DEFAULT_GRID;
    try { grid = get_option("Grid"); } catch(...) {}
    board = Board(grid);
    try {
        std::string fp = get_option("FirstPlayer");
        // board.set_side_to_move(fp);
    } catch(...) {}
}

std::string QuantumOXEngine::play_moves(const std::vector<int>& moves) {
    for (int mv : moves) board.make_move(mv);
    return format_info_string("applied " + std::to_string(static_cast<int>(moves.size())) + " moves");
}

std::string QuantumOXEngine::play_move(int move) {
    board.make_move(move);
    return format_info_string("played " + std::to_string(move));
}

// ------------------ search / go ----------------------------------------

GoResult QuantumOXEngine::go(std::optional<int> depth,
                             std::optional<int> time_ms,
                             std::optional<int> nodes) {
    Searcher::SearchResult res = searcher.search(board, depth, time_ms, nodes);

    GoResult out;
    out.raw = res;

    // Build info lines
    // for (const InfoRecord& d : res.infos) {
    //     std::ostringstream oss;
    //     // Build PV strings
    //     std::string pv_main_str, neg_pv_str, min_pv_str;
    //     for (size_t i = 0; i < d.pv.size(); ++i) { if (i) pv_main_str += " "; pv_main_str += std::to_string(d.pv[i]); }
    //     for (size_t i = 0; i < d.negamaxpv.size(); ++i) { if (i) neg_pv_str += " "; neg_pv_str += std::to_string(d.negamaxpv[i]); }
    //     for (size_t i = 0; i < d.minimaxpv.size(); ++i) { if (i) min_pv_str += " "; min_pv_str += std::to_string(d.minimaxpv[i]); }
    // 
    //     oss << "info depth " << d.depth
    //         << " seldepth " << d.seldepth
    //         << " score " << d.score
    //         << " nodes " << d.nodes
    //         << " minimaxpv " << min_pv_str
    //         << " negamaxpv " << neg_pv_str
    //         << " time " << d.time_ms
    //         << " pv " << pv_main_str;
    // 
    //     out.info_lines.push_back(oss.str());
    // }

    // Bestmove
    std::optional<int> bestmove = res.bestmove;
    std::vector<int> pv = res.pv;
    std::optional<int> ponder = std::nullopt;
    if (pv.size() > 1) ponder = pv[1];

    if (!bestmove.has_value()) {
        out.bestmove_line = format_bestmove(0, -1);
        out.bestmove = std::nullopt;
    } else {
        out.bestmove_line = format_bestmove(*bestmove, ponder.has_value() ? *ponder : -1);
        out.bestmove = bestmove;
    }

    return out;
}

// ------------------ stop -----------------------------------------------

void QuantumOXEngine::stop() {
    searcher.request_abort();
}

// ------------------ utilities ------------------------------------------

std::string QuantumOXEngine::board_state() const {
    return board.to_string();
}

std::vector<int> QuantumOXEngine::legal_moves() const {
    return board.legal_moves();
}

} // namespace QuantumOX
