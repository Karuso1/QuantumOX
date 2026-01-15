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

#ifndef BOARD_H
#define BOARD_H

#include <vector>
#include <string>
#include <optional>
#include <cstdint>

#include "constants.h"
#include "utils.h"

namespace QuantumOX {

    class Board {
    public:
        // ------------------ constructors / initialization ----------------------
        explicit Board(const std::string& grid_spec = DEFAULT_GRID);
    
        // ------------------ move / state management ---------------------------
        std::vector<int> legal_moves() const;
        void make_move(int move);
        void unmake_move(int move);
    
        // ------------------ game status ---------------------------------------
        bool is_win(const std::string& player) const;
        bool is_draw() const;
    
        // ------------------ evaluation ----------------------------------------
        int evaluate(char player) const;
        std::vector<int> get_dims() const { return dims; }
    
        // ------------------ zobrist hashing -----------------------------------
        void init_zobrist(uint64_t seed = 0);
        uint64_t zobrist_key() const;
    
        // ------------------ utilities -----------------------------------------
        void fill_from_list(const std::vector<int>& moves);
        bool validate_tttn(const std::string& tttn_str) const;
        void load_tttn(const std::string& tttn_str);
        void reset();
        std::string to_string() const;
        std::string get_side_to_move() const;
    
    private:
        std::string grid_spec;
        std::vector<char> cells;
        std::vector<int> dims;
        std::string side_to_move;
        std::vector<int> move_stack;
        std::optional<std::vector<std::vector<uint64_t>>> zobrist_table;
        std::vector<std::vector<int>> win_lines;
    
        // Bitboard representation for fast operations
        int board_size;
        std::vector<uint64_t> bitboard_x;
        std::vector<uint64_t> bitboard_o;
        std::vector<uint64_t> occupied;
        std::vector<std::vector<uint64_t>> row_masks;
        std::vector<std::vector<uint64_t>> col_masks;
        std::vector<uint64_t> diag1_mask;
        std::vector<uint64_t> diag2_mask;
    
        // ------------------ internal helpers ----------------------------------
        std::vector<std::vector<int>> generate_win_lines() const;
        int coords_to_index(const std::vector<int>& coords) const;
        void update_bitboards(int move, const std::string& player);
        bool check_win_bitboard(const std::vector<uint64_t>& bb) const;
        int bit_position(int move) const;
        int move_from_bit(int bit) const;
    };

} // namespace QuantumOX

#endif // BOARD_H
