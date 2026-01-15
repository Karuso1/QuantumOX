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

#include "board.h"

#include <sstream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <numeric>
#include <cctype>

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

    // parse a single row token string (like "1X1" or "X2" or "3") to actual symbols
    // supports digits (repeat empties), '.' as one empty, X/x, O/o.
    static void parse_row_to_symbols(const std::string& row,
                                     int expected_cols,
                                     std::vector<std::string>& out_symbols) {
        out_symbols.clear();
        out_symbols.reserve(static_cast<size_t>(expected_cols));

        std::string s_empty = symbol_to_string_auto(SYMBOL_EMPTY);
        std::string s_x = symbol_to_string_auto(SYMBOL_X);
        std::string s_o = symbol_to_string_auto(SYMBOL_O);

        size_t i = 0;
        while (i < row.size()) {
            char ch = row[i];
            if (std::isdigit(static_cast<unsigned char>(ch))) {
                // read full number
                long long val = 0;
                while (i < row.size() && std::isdigit(static_cast<unsigned char>(row[i]))) {
                    val = val * 10 + (row[i] - '0');
                    ++i;
                    // avoid absurd huge counts:
                    if (val > 1000000) throw std::runtime_error("TTTN numeric run too large");
                }
                if (val < 0) throw std::runtime_error("negative repeat in TTTN");
                for (long long k = 0; k < val; ++k) out_symbols.push_back(s_empty);
            } else {
                // symbol
                if (ch == '.' || ch == '_') {
                    out_symbols.push_back(s_empty);
                } else if (ch == 'X' || ch == 'x') {
                    out_symbols.push_back(s_x);
                } else if (ch == 'O' || ch == 'o') {
                    out_symbols.push_back(s_o);
                } else {
                    // allow other whitespace/safe separators (shouldn't appear)
                    if (std::isspace(static_cast<unsigned char>(ch))) {
                        ++i;
                        continue;
                    }
                    // unknown char: error
                    std::ostringstream oss;
                    oss << "Unknown character '" << ch << "' in TTTN row";
                    throw std::runtime_error(oss.str());
                }
                ++i;
            }
            if (static_cast<int>(out_symbols.size()) > expected_cols)
                throw std::runtime_error("Row expands to more columns than expected in TTTN");
        }

        if (static_cast<int>(out_symbols.size()) != expected_cols) {
            std::ostringstream oss;
            oss << "Row length mismatch in TTTN: expected " << expected_cols
                << " but got " << out_symbols.size();
            throw std::runtime_error(oss.str());
        }
    }

    // ------------------ constructors / initialization ---------------------------
    Board::Board(const std::string& grid_spec_in)
        : grid_spec(grid_spec_in),
          side_to_move(symbol_to_string_auto(SYMBOL_X)) // default start
    {
        // dims
        dims = parse_grid_spec(grid_spec); // expects vector<int> like {layers, rows, cols} or {rows, cols}
        if (dims.empty()) throw std::runtime_error("Invalid grid_spec, no dims parsed");

        // cells
        int total = product(dims);
        cells.assign(static_cast<size_t>(total), SYMBOL_EMPTY);

        // zobrist
        init_zobrist(0);

        // precompute win lines
        win_lines = generate_win_lines();

        // initialize bitboards
        board_size = product(dims);
        int num_words = (board_size + 63) / 64;
        bitboard_x.assign(static_cast<size_t>(num_words), 0ULL);
        bitboard_o.assign(static_cast<size_t>(num_words), 0ULL);
        occupied.assign(static_cast<size_t>(num_words), 0ULL);

        // precompute masks for 2D square grids
        if (dims.size() == 2 && dims[0] == dims[1]) {
            int N = dims[0];
            row_masks.assign(static_cast<size_t>(N), std::vector<uint64_t>(static_cast<size_t>(num_words), 0ULL));
            col_masks.assign(static_cast<size_t>(N), std::vector<uint64_t>(static_cast<size_t>(num_words), 0ULL));
            diag1_mask.assign(static_cast<size_t>(num_words), 0ULL);
            diag2_mask.assign(static_cast<size_t>(num_words), 0ULL);
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < N; ++c) {
                    int pos = r * N + c;
                    int word = pos / 64;
                    uint64_t bit = 1ULL << (pos % 64);
                    row_masks[static_cast<size_t>(r)][static_cast<size_t>(word)] |= bit;
                    col_masks[static_cast<size_t>(c)][static_cast<size_t>(word)] |= bit;
                    if (r == c) diag1_mask[static_cast<size_t>(word)] |= bit;
                    if (r + c == N - 1) diag2_mask[static_cast<size_t>(word)] |= bit;
                }
            }
        }
    }

    // ------------------ move / state management --------------------------------
    std::vector<int> Board::legal_moves() const {
        if (dims.size() == 2 && dims[0] == dims[1]) {
            std::vector<int> moves;
            moves.reserve(static_cast<size_t>(board_size));
            for (int pos = 0; pos < board_size; ++pos) {
                int word = pos / 64;
                uint64_t bit = 1ULL << (pos % 64);
                if ((occupied[static_cast<size_t>(word)] & bit) == 0) moves.push_back(pos + 1);
            }
            return moves;
        } else {
            std::vector<int> moves;
            moves.reserve(cells.size());
            for (size_t i = 0; i < cells.size(); ++i) {
                if (cells[i] == SYMBOL_EMPTY) moves.push_back(static_cast<int>(i) + 1);
            }
            return moves;
        }
    }

    void Board::make_move(int move) {
        if (move < 1) throw std::runtime_error("Move index must be >= 1");
        size_t idx = static_cast<size_t>(move - 1);
        if (idx >= cells.size()) throw std::runtime_error("Move " + std::to_string(move) + " out of range for board");
        if (cells[idx] != SYMBOL_EMPTY) throw std::runtime_error("Cell " + std::to_string(move) + " is not empty");

        // place
        cells[idx] = side_to_move[0];
        move_stack.push_back(move);

        // update bitboards
        update_bitboards(move, side_to_move);

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
        cells[idx] = SYMBOL_EMPTY;

        // update bitboards (clear)
        update_bitboards(move, "");

        // flip side back
        std::string s_x = symbol_to_string_auto(SYMBOL_X);
        std::string s_o = symbol_to_string_auto(SYMBOL_O);
        side_to_move = (side_to_move == s_x) ? s_o : s_x;
    }

    // ------------------ game status --------------------------------------------
    bool Board::is_win(const std::string& player) const {
        if (dims.size() == 2 && dims[0] == dims[1]) {
            const auto& bb = (player == symbol_to_string_auto(SYMBOL_X)) ? bitboard_x : bitboard_o;
            return check_win_bitboard(bb);
        } else {
            for (const auto & line : win_lines) {
                bool all_match = true;
                for (int idx1 : line) {
                    size_t pos = static_cast<size_t>(idx1 - 1);
                    if (pos >= cells.size() || cells[pos] != player[0]) {
                        all_match = false;
                        break;
                    }
                }
                if (all_match) return true;
            }
            return false;
        }
    }

    bool Board::is_draw() const {
        if (is_win(symbol_to_string_auto(SYMBOL_X)) || is_win(symbol_to_string_auto(SYMBOL_O)))
            return false;
        if (dims.size() == 2 && dims[0] == dims[1]) {
            uint64_t full = ~0ULL;
            for (size_t w = 0; w < occupied.size(); ++w) {
                if (occupied[w] != full) return false;
            }
            return true;
        } else {
            return std::all_of(cells.begin(), cells.end(), [](char c){ return c == SYMBOL_EMPTY; });
        }
    }

    // ------------------ evaluation ---------------------------------------------
    int Board::evaluate(char player) const {
        char opp = (player == SYMBOL_X) ? SYMBOL_O : SYMBOL_X;

        int score = 0;
        for (const auto & line : win_lines) {
            std::vector<char> marks;
            marks.reserve(line.size());
            for (int idx1 : line) {
                marks.push_back(cells[static_cast<size_t>(idx1 - 1)]);
            }
            bool player_present = std::any_of(marks.begin(), marks.end(), [&](char m){ return m == player; });
            bool opp_present = std::any_of(marks.begin(), marks.end(), [&](char m){ return m == opp; });
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

        // Add center bonus for 2D square grids
        if (dims.size() == 2 && dims[0] == dims[1]) {
            int N = dims[0];
            int center = N / 2;
            int center_pos = center * N + center;
            if (cells[static_cast<size_t>(center_pos)] == player) score += 50;
            else if (cells[static_cast<size_t>(center_pos)] == opp) score -= 50;
            // Add corner bonus
            std::vector<int> corners = {0, N-1, (N-1)*N, N*N-1};
            for (int c : corners) {
                if (cells[static_cast<size_t>(c)] == player) score += 20;
                else if (cells[static_cast<size_t>(c)] == opp) score -= 20;
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
            if (cells[i] == SYMBOL_X) {
                h ^= (*zobrist_table)[i][0];
            } else if (cells[i] == SYMBOL_O) {
                h ^= (*zobrist_table)[i][1];
            }
        }
        if (side_to_move == symbol_to_string_auto(SYMBOL_O)) {
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
        cells.assign(cells.size(), SYMBOL_EMPTY);
        bitboard_x.assign(bitboard_x.size(), 0ULL);
        bitboard_o.assign(bitboard_o.size(), 0ULL);
        occupied.assign(occupied.size(), 0ULL);
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

    // ------------------ TTTN validation & loader --------------------------------
    // validate_tttn: basic syntax check + dims match
    bool Board::validate_tttn(const std::string& tttn_str) const {
        // Acceptable formats: (based on dims.size())
        // 1D: a token that expands to dims[0] cells (e.g. "3" or "X2X")
        // 2D: rows separated by '/', number of rows == dims[0], each expands to dims[1]
        // 3D: layers separated by '|', each layer has rows separated by '/', #layers == dims[0], #rows == dims[1], each row expands to dims[2]
        if (dims.empty()) return false;

        try {
            if (dims.size() == 1) {
                // single token
                std::vector<std::string> syms;
                parse_row_to_symbols(tttn_str, dims[0], syms);
                (void)syms;
                return true;
            } else if (dims.size() == 2) {
                // rows separated by '/'
                std::vector<std::string> rows;
                std::string cur;
                for (char ch : tttn_str) {
                    if (ch == '/') {
                        rows.push_back(cur);
                        cur.clear();
                    } else cur.push_back(ch);
                }
                rows.push_back(cur);
                if (static_cast<int>(rows.size()) != dims[0]) return false;
                for (const auto &r : rows) {
                    std::vector<std::string> syms;
                    parse_row_to_symbols(r, dims[1], syms);
                }
                return true;
            } else if (dims.size() == 3) {
                // layers separated by '|'
                std::vector<std::string> layers;
                std::string cur;
                for (char ch : tttn_str) {
                    if (ch == '|') {
                        layers.push_back(cur);
                        cur.clear();
                    } else cur.push_back(ch);
                }
                layers.push_back(cur);
                if (static_cast<int>(layers.size()) != dims[0]) return false;
                for (const auto &layer : layers) {
                    // split rows
                    std::vector<std::string> rows;
                    std::string rcur;
                    for (char ch : layer) {
                        if (ch == '/') {
                            rows.push_back(rcur);
                            rcur.clear();
                        } else rcur.push_back(ch);
                    }
                    rows.push_back(rcur);
                    if (static_cast<int>(rows.size()) != dims[1]) return false;
                    for (const auto &r : rows) {
                        std::vector<std::string> syms;
                        parse_row_to_symbols(r, dims[2], syms);
                    }
                }
                return true;
            } else {
                // not implemented for >3 dims in TTTN textual format
                return false;
            }
        } catch (...) {
            return false;
        }
    }

    // load_tttn: parse and write into cells; infers side-to-move from counts (X starts)
    void Board::load_tttn(const std::string& tttn_str) {
        if (!validate_tttn(tttn_str)) {
            throw std::runtime_error("TTTN validation failed or does not match current grid dimensions");
        }

        std::string s_empty = symbol_to_string_auto(SYMBOL_EMPTY);
        std::string s_x = symbol_to_string_auto(SYMBOL_X);
        std::string s_o = symbol_to_string_auto(SYMBOL_O);

        // reset board
        cells.assign(cells.size(), SYMBOL_EMPTY);
        move_stack.clear();

        int count_x = 0;
        int count_o = 0;

        if (dims.size() == 1) {
            std::vector<std::string> syms;
            parse_row_to_symbols(tttn_str, dims[0], syms);
            for (int i = 0; i < dims[0]; ++i) {
                cells[static_cast<size_t>(i)] = syms[static_cast<size_t>(i)][0];
                if (syms[static_cast<size_t>(i)] == s_x) ++count_x;
                else if (syms[static_cast<size_t>(i)] == s_o) ++count_o;
            }
        } else if (dims.size() == 2) {
            // split rows
            std::vector<std::string> rows;
            std::string cur;
            for (char ch : tttn_str) {
                if (ch == '/') {
                    rows.push_back(cur);
                    cur.clear();
                } else cur.push_back(ch);
            }
            rows.push_back(cur);
            int rowsN = dims[0];
            int colsN = dims[1];
            for (int r = 0; r < rowsN; ++r) {
                std::vector<std::string> syms;
                parse_row_to_symbols(rows[static_cast<size_t>(r)], colsN, syms);
                for (int c = 0; c < colsN; ++c) {
                    int idx = (r * colsN + c);
                    cells[static_cast<size_t>(idx)] = syms[static_cast<size_t>(c)][0];
                    if (syms[static_cast<size_t>(c)] == s_x) ++count_x;
                    else if (syms[static_cast<size_t>(c)] == s_o) ++count_o;
                }
            }
        } else if (dims.size() == 3) {
            // split layers by '|'
            std::vector<std::string> layers;
            std::string curL;
            for (char ch : tttn_str) {
                if (ch == '|') {
                    layers.push_back(curL);
                    curL.clear();
                } else curL.push_back(ch);
            }
            layers.push_back(curL);
            int L = dims[0];
            int R = dims[1];
            int C = dims[2];

            for (int l = 0; l < L; ++l) {
                const std::string &layer = layers[static_cast<size_t>(l)];
                // split rows by '/'
                std::vector<std::string> rows;
                std::string curR;
                for (char ch : layer) {
                    if (ch == '/') {
                        rows.push_back(curR);
                        curR.clear();
                    } else curR.push_back(ch);
                }
                rows.push_back(curR);
                for (int r = 0; r < R; ++r) {
                    std::vector<std::string> syms;
                    parse_row_to_symbols(rows[static_cast<size_t>(r)], C, syms);
                    for (int c = 0; c < C; ++c) {
                        // coords: [l, r, c]
                        std::vector<int> coords = { l, r, c };
                        int one_based_idx = coords_to_index(coords);
                        int zero_idx = one_based_idx - 1;
                        cells[static_cast<size_t>(zero_idx)] = syms[static_cast<size_t>(c)][0];
                        if (syms[static_cast<size_t>(c)] == s_x) ++count_x;
                        else if (syms[static_cast<size_t>(c)] == s_o) ++count_o;
                    }
                }
            }
        } else {
            throw std::runtime_error("TTTN loader not implemented for dims > 3");
        }

        // update bitboards
        bitboard_x.assign(bitboard_x.size(), 0ULL);
        bitboard_o.assign(bitboard_o.size(), 0ULL);
        for (int i = 0; i < board_size; ++i) {
            if (cells[static_cast<size_t>(i)] == SYMBOL_X) {
                int word = i / 64;
                bitboard_x[static_cast<size_t>(word)] |= (1ULL << (i % 64));
            } else if (cells[static_cast<size_t>(i)] == SYMBOL_O) {
                int word = i / 64;
                bitboard_o[static_cast<size_t>(word)] |= (1ULL << (i % 64));
            }
        }
        for (size_t w = 0; w < occupied.size(); ++w) {
            occupied[w] = bitboard_x[w] | bitboard_o[w];
        }

        // infer side to move: X starts first
        if (count_x == count_o) {
            side_to_move = s_x;
        } else if (count_x == count_o + 1) {
            side_to_move = s_o;
        } else {
            std::ostringstream oss;
            oss << "Invalid piece counts in TTTN: X=" << count_x << " O=" << count_o;
            throw std::runtime_error(oss.str());
        }

        // regenerate derived data
        win_lines = generate_win_lines();
        init_zobrist(0); // re-init (deterministic seed 0); change seed behavior if desired
    }

    void Board::update_bitboards(int move, const std::string& player) {
        int pos = move - 1;
        int word = pos / 64;
        uint64_t bit = 1ULL << (pos % 64);
        if (player == symbol_to_string_auto(SYMBOL_X)) {
            bitboard_x[static_cast<size_t>(word)] |= bit;
        } else if (player == symbol_to_string_auto(SYMBOL_O)) {
            bitboard_o[static_cast<size_t>(word)] |= bit;
        } else {
            // clear
            bitboard_x[static_cast<size_t>(word)] &= ~bit;
            bitboard_o[static_cast<size_t>(word)] &= ~bit;
        }
        occupied[static_cast<size_t>(word)] = bitboard_x[static_cast<size_t>(word)] | bitboard_o[static_cast<size_t>(word)];
    }

    bool Board::check_win_bitboard(const std::vector<uint64_t>& bb) const {
        if (dims.size() != 2 || dims[0] != dims[1]) return false;
        int N = dims[0];
        for (int i = 0; i < N; ++i) {
            bool row_win = true;
            bool col_win = true;
            for (size_t w = 0; w < row_masks[static_cast<size_t>(i)].size(); ++w) {
                if ((bb[w] & row_masks[static_cast<size_t>(i)][w]) != row_masks[static_cast<size_t>(i)][w]) row_win = false;
                if ((bb[w] & col_masks[static_cast<size_t>(i)][w]) != col_masks[static_cast<size_t>(i)][w]) col_win = false;
            }
            if (row_win || col_win) return true;
        }
        bool d1_win = true;
        bool d2_win = true;
        for (size_t w = 0; w < diag1_mask.size(); ++w) {
            if ((bb[w] & diag1_mask[w]) != diag1_mask[w]) d1_win = false;
            if ((bb[w] & diag2_mask[w]) != diag2_mask[w]) d2_win = false;
        }
        return d1_win || d2_win;
    }

    int Board::bit_position(int move) const {
        return move - 1;
    }

    int Board::move_from_bit(int bit) const {
        return bit + 1;
    }

} // namespace QuantumOX
