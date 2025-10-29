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

#include "board.h"

#include <sstream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <numeric>

namespace QuantumOX {

    // ------------------ helpers --------------------------------------------------
    // convert symbol (char or string) to string by streaming it (works for both)
    static std::string symbol_to_string_auto(const auto& sym) {
        std::ostringstream oss;
        oss << sym;
        return oss.str();
    }
    
    static int product(const std::vector<int>& nums) {
        if (nums.empty()) return 0;
        return std::accumulate(nums.begin(), nums.end(), 1, std::multiplies<int>());
    }
    
    // ------------------ constructors / initialization ---------------------------
    Board::Board(const std::string& grid_spec_in)
        : grid_spec(grid_spec_in),
          side_to_move(symbol_to_string_auto(SYMBOL_X)) // default start
    {
        // dims
        dims = parse_grid_spec(grid_spec); // expects vector<int>
        // cells
        int total = product(dims);
        std::string s_empty = symbol_to_string_auto(SYMBOL_EMPTY);
        cells.assign(total, s_empty);
    
        // zobrist
        init_zobrist(0);
    
        // precompute win lines
        win_lines = generate_win_lines();
    }
    
    // ------------------ move / state management --------------------------------
    std::vector<int> Board::legal_moves() const {
        std::string s_empty = symbol_to_string_auto(SYMBOL_EMPTY);
        std::vector<int> moves;
        moves.reserve(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            if (cells[i] == s_empty) moves.push_back(static_cast<int>(i) + 1);
        }
        return moves;
    }
    
    void Board::make_move(int move) {
        if (move < 1) throw std::runtime_error("Move index must be >= 1");
        size_t idx = static_cast<size_t>(move - 1);
        if (idx >= cells.size()) throw std::runtime_error("Move " + std::to_string(move) + " out of range for board");
        std::string s_empty = symbol_to_string_auto(SYMBOL_EMPTY);
        if (cells[idx] != s_empty) throw std::runtime_error("Cell " + std::to_string(move) + " is not empty");
    
        // place
        cells[idx] = side_to_move;
        move_stack.push_back(move);
    
        // flip side
        std::string s_x = symbol_to_string_auto(SYMBOL_X);
        std::string s_o = symbol_to_string_auto(SYMBOL_O);
        side_to_move = (side_to_move == s_x) ? s_o : s_x;
    }
    
    void Board::unmake_move(int move) {
        if (move_stack.empty()) throw std::runtime_error("Unmake called but move stack is empty");
        int last = move_stack.back();
        move_stack.pop_back();
        if (last != move) {
            throw std::logic_error("Unmake called with " + std::to_string(move) + " but last move was " + std::to_string(last));
        }
        size_t idx = static_cast<size_t>(move - 1);
        if (idx >= cells.size()) throw std::runtime_error("Move " + std::to_string(move) + " out of range for board");
        std::string s_empty = symbol_to_string_auto(SYMBOL_EMPTY);
        cells[idx] = s_empty;
    
        // flip side back
        std::string s_x = symbol_to_string_auto(SYMBOL_X);
        std::string s_o = symbol_to_string_auto(SYMBOL_O);
        side_to_move = (side_to_move == s_x) ? s_o : s_x;
    }
    
    // ------------------ game status --------------------------------------------
    bool Board::is_win(const std::string& player) const {
        for (const auto & line : win_lines) {
            bool all_match = true;
            for (int idx1 : line) {
                size_t pos = static_cast<size_t>(idx1 - 1);
                if (pos >= cells.size() || cells[pos] != player) {
                    all_match = false;
                    break;
                }
            }
            if (all_match) return true;
        }
        return false;
    }
    
    bool Board::is_draw() const {
        std::string s_empty = symbol_to_string_auto(SYMBOL_EMPTY);
        if (is_win(symbol_to_string_auto(SYMBOL_X)) || is_win(symbol_to_string_auto(SYMBOL_O)))
            return false;
        return std::all_of(cells.begin(), cells.end(), [&](const std::string &c){ return c != s_empty; });
    }
    
    // ------------------ evaluation ---------------------------------------------
    int Board::evaluate(const std::string& player) const {
        std::string s_x = symbol_to_string_auto(SYMBOL_X);
        std::string s_o = symbol_to_string_auto(SYMBOL_O);
        std::string opp = (player == s_x) ? s_o : s_x;
    
        int score = 0;
        for (const auto & line : win_lines) {
            std::vector<std::string> marks;
            marks.reserve(line.size());
            for (int idx1 : line) {
                marks.push_back(cells[static_cast<size_t>(idx1 - 1)]);
            }
            bool player_present = std::any_of(marks.begin(), marks.end(), [&](const std::string &m){ return m == player; });
            bool opp_present = std::any_of(marks.begin(), marks.end(), [&](const std::string &m){ return m == opp; });
            if (player_present && opp_present) continue;
            if (!player_present && !opp_present) {
                score += 1;
            } else if (player_present && !opp_present) {
                int cnt = static_cast<int>(std::count(marks.begin(), marks.end(), player));
                score += (cnt * cnt) * 10;
            } else if (opp_present && !player_present) {
                int cnt = static_cast<int>(std::count(marks.begin(), marks.end(), opp));
                score -= (cnt * cnt) * 10;
            }
        }
        return score;
    }
    
    // ------------------ zobrist hashing ----------------------------------------
    void Board::init_zobrist(uint64_t seed) {
        std::mt19937_64 rng(static_cast<uint64_t>(seed));
        std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
        int total = static_cast<int>(cells.size());
        zobrist_table = std::vector<std::vector<uint64_t>>(static_cast<size_t>(total), std::vector<uint64_t>(2));
        for (int i = 0; i < total; ++i) {
            (*zobrist_table)[static_cast<size_t>(i)][0] = dist(rng);
            (*zobrist_table)[static_cast<size_t>(i)][1] = dist(rng);
        }
    }
    
    uint64_t Board::zobrist_key() const {
        if (!zobrist_table.has_value()) {
            // non-const init fallback (shouldn't normally happen)
            const_cast<Board*>(this)->init_zobrist(0);
        }
        uint64_t h = 0;
        std::string s_x = symbol_to_string_auto(SYMBOL_X);
        std::string s_o = symbol_to_string_auto(SYMBOL_O);
        for (size_t i = 0; i < cells.size(); ++i) {
            if (cells[i] == s_x) {
                h ^= (*zobrist_table)[i][0];
            } else if (cells[i] == s_o) {
                h ^= (*zobrist_table)[i][1];
            }
        }
        if (side_to_move == s_o) {
            h ^= 0xF00DF00DCAFEBABEULL;
        }
        return h;
    }
    
    // ------------------ win-line generation ------------------------------------
    std::vector<std::vector<int>> Board::generate_win_lines() const {
        std::vector<std::vector<int>> out;
        int N = static_cast<int>(dims.size());
        if (N == 0) return out;
    
        // determine L (win length)
        int L;
        bool all_eq = std::all_of(dims.begin(), dims.end(), [&](int d){ return d == dims[0]; });
        if (all_eq) L = dims[0];
        else L = *std::min_element(dims.begin(), dims.end());
    
        // build directions: all vectors in {-1,0,1}^N except zero vector
        // we'll represent directions as vector<int> of length N with values -1,0,1
        std::vector<std::vector<int>> directions;
        int dir_count = 1;
        for (int i = 0; i < N; ++i) dir_count *= 3; // 3^N
        directions.reserve(dir_count);
        for (int code = 0; code < dir_count; ++code) {
            int x = code;
            std::vector<int> d(N);
            bool all_zero = true;
            for (int i = 0; i < N; ++i) {
                int rem = x % 3;
                x /= 3;
                int val = (rem - 1); // rem:0->-1,1->0,2->1
                d[i] = val;
                if (val != 0) all_zero = false;
            }
            if (all_zero) continue;
            // canonicalize: first non-zero component must be positive
            bool ok = false;
            for (int i = 0; i < N; ++i) {
                if (d[i] != 0) {
                    if (d[i] > 0) ok = true;
                    break;
                }
            }
            if (ok) directions.push_back(d);
        }
    
        // generate all start coordinates (cartesian product of ranges)
        // We'll implement an odometer-style loop over dims
        std::vector<int> start(N, 0);
        bool finished = false;
        while (!finished) {
            // for each direction, check whether a line of length L fits
            for (const auto &d : directions) {
                bool fits = true;
                std::vector<int> line_indices;
                line_indices.reserve(static_cast<size_t>(L));
                for (int k = 0; k < L; ++k) {
                    std::vector<int> coord(N);
                    for (int i = 0; i < N; ++i) {
                        coord[i] = start[i] + k * d[i];
                    }
                    // check bounds
                    bool inb = true;
                    for (int i = 0; i < N; ++i) {
                        if (coord[i] < 0 || coord[i] >= dims[static_cast<size_t>(i)]) {
                            inb = false;
                            break;
                        }
                    }
                    if (!inb) {
                        fits = false;
                        break;
                    }
                    // convert coords -> 1-based index (row-major)
                    int idx = 0;
                    for (int i = 0; i < N; ++i) {
                        idx = idx * dims[static_cast<size_t>(i)] + coord[static_cast<size_t>(i)];
                    }
                    line_indices.push_back(idx + 1);
                }
                if (fits) out.push_back(line_indices);
            }
        
            // increment odometer
            int pos = N - 1;
            while (pos >= 0) {
                start[pos] += 1;
                if (start[pos] < dims[static_cast<size_t>(pos)]) break;
                start[pos] = 0;
                --pos;
            }
            if (pos < 0) finished = true;
        }
    
        return out;
    }
    
    // ------------------ utilities ----------------------------------------------
    int Board::coords_to_index(const std::vector<int>& coords) const {
        if (coords.size() != dims.size()) throw std::runtime_error("coords length must match dims length");
        int idx = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            if (coords[i] < 0 || coords[i] >= dims[i]) throw std::runtime_error("coordinate out of range");
            idx = idx * dims[i] + coords[i];
        }
        return idx + 1;
    }
    
    void Board::fill_from_list(const std::vector<int>& moves) {
        for (int mv : moves) make_move(mv);
    }
    
    void Board::reset() {
        std::string s_empty = symbol_to_string_auto(SYMBOL_EMPTY);
        cells.assign(cells.size(), s_empty);
        move_stack.clear();
        side_to_move = symbol_to_string_auto(SYMBOL_X);
    }
    
    std::string Board::to_string() const {
        std::ostringstream oss;
        if (dims.size() == 2) {
            int rows = dims[0];
            int cols = dims[1];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    if (c) oss << ' ';
                    oss << cells[static_cast<size_t>(r * cols + c)];
                }
                if (r != rows - 1) oss << '\n';
            }
            return oss.str();
        }
        for (size_t i = 0; i < cells.size(); ++i) {
            if (i) oss << ' ';
            oss << (i + 1) << ':' << cells[i];
        }
        return oss.str();
    }

    std::string Board::get_side_to_move() const {
        return side_to_move;
    }

} // namespace QuantumOX
