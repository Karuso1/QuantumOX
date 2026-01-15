/* 
 * QuantumOX, a Tic Tac Toe engine supporting UTTTI.
 * Copyright (C) 2025-2026 Kartik Pawar
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

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <sstream>
#include <cctype>
#include <stdexcept>
#include <algorithm>
#include "constants.h"

namespace QuantumOX {

    // --- command tokenization ----------------------------------------------------
    inline std::vector<std::string> tokenize_command(const std::string& line) {
        std::vector<std::string> tokens;
        std::istringstream stream(line);
        std::string token;
        char c;
        bool in_quotes = false;
        std::string temp;
    
        for (size_t i = 0; i < line.size(); ++i) {
            c = line[i];
            if (c == '"') {
                in_quotes = !in_quotes;
                if (!in_quotes && !temp.empty()) {
                    tokens.push_back(temp);
                    temp.clear();
                }
            } else if (std::isspace(c) && !in_quotes) {
                if (!temp.empty()) {
                    tokens.push_back(temp);
                    temp.clear();
                }
            } else {
                temp += c;
            }
        }
        if (!temp.empty()) tokens.push_back(temp);
        return tokens;
    }
    
    // --- setoption parsing ------------------------------------------------------
    inline std::pair<std::string, std::string> parse_setoption(const std::vector<std::string>& tokens) {
        std::string name, value;
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string tok_lower = tokens[i];
            std::transform(tok_lower.begin(), tok_lower.end(), tok_lower.begin(), ::tolower);
            if (tok_lower == "name" && i + 1 < tokens.size()) name = tokens[i + 1];
            if (tok_lower == "value" && i + 1 < tokens.size()) value = tokens[i + 1];
        }
        return {name, value};
    }
    
    // --- grid helpers -----------------------------------------------------------
    inline std::vector<int> default_grid_dims() {
        return parse_grid_spec(DEFAULT_GRID);
    }
    
    inline std::vector<int> index_to_coords(int index, const std::vector<int>& dims) {
        if (index < 1) throw std::runtime_error("Index must be >= 1");
        int idx = index - 1;
        std::vector<int> coords;
        for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
            coords.push_back(idx % *it);
            idx /= *it;
        }
        if (idx != 0) throw std::runtime_error("Index out of range for grid dimensions");
        std::reverse(coords.begin(), coords.end());
        return coords;
    }
    
    inline int coords_to_index(const std::vector<int>& coords, const std::vector<int>& dims) {
        if (coords.size() != dims.size()) throw std::runtime_error("coords length must match dims length");
        int idx = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            if (coords[i] < 0 || coords[i] >= dims[i])
                throw std::runtime_error("coordinate out of range");
            idx = idx * dims[i] + coords[i];
        }
        return idx + 1;
    }
    
    // --- move parsing -----------------------------------------------------------
    inline int parse_move_token(const std::string& tok) {
        std::string trimmed;
        std::remove_copy_if(tok.begin(), tok.end(), std::back_inserter(trimmed), ::isspace);
    
        if (std::all_of(trimmed.begin(), trimmed.end(), ::isdigit)) {
            return std::stoi(trimmed);
        }
        auto comma_pos = trimmed.find(',');
        if (comma_pos != std::string::npos) {
            std::vector<int> coords;
            std::istringstream ss(trimmed);
            std::string part;
            while (std::getline(ss, part, ',')) coords.push_back(std::stoi(part));
            return coords_to_index(coords, default_grid_dims());
        }
        throw std::runtime_error("Unrecognized move token: " + tok);
    }
    
    // --- formatting helpers for UTTTI output -----------------------------------
    inline std::string format_info_line(int depth = -1, int seldepth = -1, int score_cp = -1,
                                        int nodes = -1, const std::vector<int>& pv = {}) {
        std::string result = "info";
        if (depth != -1) result += " depth " + std::to_string(depth);
        if (seldepth != -1) result += " seldepth " + std::to_string(seldepth);
        if (score_cp != -1) result += " score cp " + std::to_string(score_cp);
        if (nodes != -1) result += " nodes " + std::to_string(nodes);
        if (!pv.empty()) {
            result += " pv";
            for (auto m : pv) result += " " + std::to_string(m);
        }
        return result;
    }
    
    inline std::string format_info_string(const std::string& msg) {
        return std::string(INFO_STRING_PREFIX) + " " + msg;
    }
    
    inline std::string format_bestmove(int move, int ponder = -1) {
        std::string result = "bestmove " + std::to_string(move);
        if (ponder != -1) result += " ponder " + std::to_string(ponder);
        return result;
    }
    
    // --- board pretty-printer ---------------------------------------------------
    inline std::string pretty_print_board(const std::vector<std::string>& cells, const std::vector<int>& dims) {
        if (dims.size() == 2) {
            int rows = dims[0], cols = dims[1];
            if ((int)cells.size() != rows * cols)
                throw std::runtime_error("cells length doesn't match dims");
            std::string out;
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    out += cells[r * cols + c];
                    if (c != cols - 1) out += " ";
                }
                if (r != rows - 1) out += "\n";
            }
            return out;
        }
        std::string out;
        for (size_t i = 0; i < cells.size(); ++i)
            out += std::to_string(i + 1) + ":" + cells[i] + " ";
        if (!out.empty()) out.pop_back(); // remove last space
        return out;
    }
    
    // --- tiny helpers -----------------------------------------------------------
    inline std::vector<std::string> empty_grid_cells(const std::vector<int>& dims) {
        int total = 1;
        for (auto d : dims) total *= d;
        return std::vector<std::string>(total, std::string(1, SYMBOL_EMPTY));
    }

}

#endif // UTILS_H
