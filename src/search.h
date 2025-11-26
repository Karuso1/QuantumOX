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

#ifndef SEARCH_H
#define SEARCH_H

#include <cstdint>
#include <functional>
#include <vector>
#include <unordered_map>
#include <optional>
#include <chrono>
#include <string>
#include <memory> // for std::shared_ptr

#include "board.h"

namespace QuantumOX {

    // forward-declare ThreadPool so search.h doesn't need to pull in the
    // thread-pool implementation. The actual ThreadPool can live in search.cpp
    // or a dedicated threadpool.h and be defined in the QuantumOX namespace.
    class ThreadPool;

    // -------------------- Transposition Table Flag & Entry --------------------
    enum class TTFlag {
        EXACT,
        LOWER,
        UPPER
    };
    
    struct TTEntry {
        uint64_t key;
        int depth;
        int score;
        TTFlag flag;
        std::optional<int> best_move;
    };
    
    // -------------------- InfoRecord & SearchResult ---------------------------
    struct InfoRecord {
        int depth{}; 
        int seldepth{}; 
        int score{}; 
        uint64_t nodes{}; 
        uint64_t time_ms{}; 
        uint64_t nps{};
        std::vector<int> negamaxpv;
        std::vector<int> minimaxpv;
        std::vector<int> pv;
    };
    
    class Searcher {
    public:
        using InfoCallback = std::function<void(const InfoRecord&)>;
        Searcher();
    
        // ----------------- Public API -----------------------------------
        struct SearchResult {
            std::optional<int> bestmove;
            int score{};
            std::vector<int> pv;
            int nodes{};
            std::vector<InfoRecord> infos;
        };
    
        // Main search entry (iterative deepening + negamax/minimax)
        SearchResult search(Board& board,
                            std::optional<int> max_depth = std::nullopt,
                            std::optional<int> time_ms = std::nullopt,
                            std::optional<int> nodes_limit = std::nullopt);
        
        // Quick move ordering heuristic (makes/unmakes move safely)
        int quick_move_score(Board& board, int move, const std::string& for_player);
        
        // ----------------- Abort request -----------------------
        void request_abort();

        // ----------------- Internal helpers -----------------------------
        // Note: ThreadPool is forward-declared above; search.cpp provides the definition.
        int negamax_root(Board& board, int depth, int alpha, int beta, int root_depth, std::shared_ptr<ThreadPool> pool);
        int negamax(Board& board, int depth, int alpha, int beta, int root_depth);
        
        int minimax_root(Board& board, int depth, int alpha, int beta, int root_depth, std::shared_ptr<ThreadPool> pool);
        int minimax(Board& board, int depth, int alpha, int beta, const std::string& root_player, int root_depth);
        
        std::vector<int> build_pv(Board& board);
        std::vector<int> build_pv_for_root(Board& board, const std::string& root_player);
        
        int evaluate_terminal(Board& board);
        int evaluate_terminal_or_heuristic(Board& board);
        int evaluate_for_root(Board& board, const std::string& root_player);
        
        // compute key (zobrist if available, else hash of board string)
        uint64_t key(Board& board);
        
        // store into the two TT maps used by the .cpp
        void store_tt_plain(uint64_t key, int depth, int score, TTFlag flag, std::optional<int> best_move);
        void store_tt_root(uint64_t root_key, int depth, int score, TTFlag flag, std::optional<int> best_move);
        
        // ----------------- Time & node management -----------------------
        bool time_exceeded() const;
        bool nodes_exceeded() const;
        bool should_abort() const;
        int elapsed_ms() const;

        // ----------------- Data members ---------------------------------
        // Plain-key transposition table (key = zobrist or hashed board)
        std::unordered_map<uint64_t, TTEntry> tt_plain;
        
        // Root-keyed transposition table (key = mix(zobrist, root_player))
        std::unordered_map<uint64_t, TTEntry> tt_root;

        int nodes{};
        std::chrono::steady_clock::time_point start_time;
        std::optional<double> time_limit; // seconds
        std::optional<int> node_limit;
        bool abort_flag{false};

        // ----------------- Selective depth tracking -----------------------
        // Real seldepth tracking: increment on every recursion entry (push_ply)
        // and decrement on exit (pop_ply). max_seldepth records the deepest
        // ply visited during an iterative-deepening iteration.
        void push_ply();  // increments current_seldepth and updates max_seldepth
        void pop_ply();   // decrements current_seldepth safely
        int max_seldepth{0};
        int current_seldepth{0};
        
        // heuristic helpers (killer/history)
        std::unordered_map<int, std::vector<int>> killer_moves; // depth -> moves
        std::unordered_map<int, int> history; // move -> score
    };
    
     // -------------------- Convenience wrapper ------------------------
    Searcher::SearchResult search_position(Board& board,
                                           std::optional<int> max_depth = std::nullopt,
                                           std::optional<int> time_ms = std::nullopt,
                                           std::optional<int> nodes_limit = std::nullopt);

} // namespace QuantumOX

#endif // SEARCH_H
