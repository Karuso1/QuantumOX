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

#include "search.h"
#include "constants.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

using namespace std::chrono;

namespace QuantumOX {

    // --- small helpers --------------------------------------------------------
    static uint64_t make_root_key(uint64_t key, char root_char) {
        return key ^ (static_cast<uint64_t>(static_cast<unsigned char>(root_char)) << 56) ^
               0x9e3779b97f4a7c15ULL;
    }
    
    static uint64_t hash_string_fallback(const std::string& s) {
        return static_cast<uint64_t>(std::hash<std::string>{}(s));
    }
    
    // --- Searcher implementation ---------------------------------------------
    
    Searcher::Searcher()
        : nodes(0),
          start_time(),
          time_limit(std::nullopt),
          node_limit(std::nullopt),
          abort_flag(false) {}
    
    int Searcher::quick_move_score(Board& board, int mv, const std::string& for_player) {
        try {
            board.make_move(mv);
        } catch (...) {
            return 0;
        }
        // ensure unmake
        try {
            int val = 0;
            try {
                val = board.evaluate(for_player);
            } catch (...) {
                val = 0;
            }
            board.unmake_move(mv);
            return val;
        } catch (...) {
            // Attempt to unmake but ignore failures
            try { board.unmake_move(mv); } catch(...) {}
            return 0;
        }
    }
    
    Searcher::SearchResult Searcher::search(Board& board,
                                           std::optional<int> max_depth_opt,
                                           std::optional<int> time_ms_opt,
                                           std::optional<int> nodes_limit_opt) {
        int max_depth = max_depth_opt.value_or(DEFAULT_MAX_DEPTH);
                                        
        nodes = 0;
        start_time = std::chrono::steady_clock::now();
        time_limit = time_ms_opt ? std::optional<double>(*time_ms_opt / 1000.0) : std::nullopt;
        node_limit = nodes_limit_opt;
        abort_flag = false;
                                        
        std::optional<int> best_move = std::nullopt;
        int best_score = 0;
        std::vector<int> best_pv;
        std::vector<InfoRecord> infos;
                                        
        uint64_t key_plain = key(board);
        std::string root_player = board.get_side_to_move(); // requires accessor in Board
                                        
        const int ASP_WINDOW = 50;
        int prev_neg_score = 0;
        int prev_min_score = 0;
                                        
        for (int depth = 1; depth <= max_depth; ++depth) {
            if (should_abort()) break;

            // ---------------- NEGAMAX pass with aspiration window ----------------
            int nodes_before = nodes;
            int alpha, beta;
            if (depth > 1) {
                alpha = prev_neg_score - ASP_WINDOW;
                beta = prev_neg_score + ASP_WINDOW;
            } else {
                alpha = -10000000;
                beta = 10000000;
            }
        
            int neg_score = negamax_root(board, depth, alpha, beta);
        
            if (depth > 1 && (neg_score <= alpha || neg_score >= beta)) {
                alpha = -10000000; beta = 10000000;
                neg_score = negamax_root(board, depth, alpha, beta);
            }
        
            int neg_nodes = nodes - nodes_before;
            prev_neg_score = neg_score;
        
            if (abort_flag) break;
        
            // ---------------- negamax PV & move ----------------
            std::optional<int> neg_move = std::nullopt;
            std::vector<int> neg_pv;
            {
                auto it = tt_plain.find(key_plain);
                if (it != tt_plain.end()) neg_move = it->second.best_move;
            }
            try { neg_pv = build_pv(board); } 
            catch(...) { if (neg_move) neg_pv = {*neg_move}; }
        
            if (should_abort()) break;
        
            // ---------------- MINIMAX pass ----------------
            int nodes_before_min = nodes;
            if (depth > 1) {
                alpha = prev_min_score - ASP_WINDOW;
                beta = prev_min_score + ASP_WINDOW;
            } else {
                alpha = -10000000; beta = 10000000;
            }
        
            int min_score = minimax_root(board, depth, alpha, beta);
            if (depth > 1 && (min_score <= alpha || min_score >= beta)) {
                alpha = -10000000; beta = 10000000;
                min_score = minimax_root(board, depth, alpha, beta);
            }
            int min_nodes = nodes - nodes_before_min;
            prev_min_score = min_score;
        
            if (abort_flag) break;
        
            std::optional<int> min_move = std::nullopt;
            std::vector<int> min_pv;
            {
                uint64_t rootk = make_root_key(key_plain, root_player.empty() ? 'X' : root_player[0]);
                auto it = tt_root.find(rootk);
                if (it != tt_root.end()) min_move = it->second.best_move;
            }
            try { min_pv = build_pv_for_root(board, root_player); }
            catch(...) {
                try { min_pv = build_pv(board); } catch(...) { if (min_move) min_pv = {*min_move}; }
            }
        
            // ---------------- pick between negamax and minimax ----------------
            std::string selector = "negamax";
            int chosen_score = neg_score;
            std::optional<int> chosen_move = neg_move;
            std::vector<int> chosen_pv = neg_pv;
        
            if (min_score > neg_score) {
                selector = "minimax";
                chosen_score = min_score;
                chosen_move = min_move;
                chosen_pv = min_pv;
            } else if (min_score == neg_score && min_nodes < neg_nodes) {
                selector = "minimax";
                chosen_score = min_score;
                chosen_move = min_move;
                chosen_pv = min_pv;
            }
        
            if (chosen_move.has_value()) {
                best_move = chosen_move;
                best_score = chosen_score;
                best_pv = chosen_pv;
            }
        
            // ---------------- record info ----------------
            InfoRecord rec;
            rec.depth = depth;
            rec.seldepth = depth;
            rec.score = best_score;
            rec.nodes = nodes;
            rec.negamaxpv = neg_pv;
            rec.minimaxpv = min_pv;
            rec.time_ms = elapsed_ms();
            rec.pv = best_pv;
            infos.push_back(std::move(rec));
        
            if (time_exceeded() || nodes_exceeded()) break;
        }
    
        Searcher::SearchResult res;
        res.bestmove = best_move;
        res.score = best_score;
        res.pv = best_pv;
        res.nodes = nodes;
        res.infos = std::move(infos);
        return res;
    }
    
    // ---------- negamax root + recursion -------------------------------------
    
    int Searcher::negamax_root(Board& board, int depth, int alpha, int beta) {
        ++nodes;
        auto moves = board.legal_moves();
        if (moves.empty()) return evaluate_terminal(board);
    
        uint64_t k = key(board);
        std::optional<int> tt_move = std::nullopt;
        auto it = tt_plain.find(k);
        if (it != tt_plain.end()) tt_move = it->second.best_move;
    
        if (tt_move) {
            auto f = std::find(moves.begin(), moves.end(), *tt_move);
            if (f != moves.end()) { moves.erase(f); moves.insert(moves.begin(), *tt_move); }
        }
    
        int best_score = -10000000;
        for (int mv : moves) {
            if (should_abort()) break;
            try { board.make_move(mv); } catch(...) { continue; }
            int score = -negamax(board, depth - 1, -beta, -alpha);
            try { board.unmake_move(mv); } catch(...) {}
        
            if (score > best_score) {
                best_score = score;
                store_tt_plain(k, depth, score, TTFlag::EXACT, mv);
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) break;
        }
        return best_score;
    }
    
    int Searcher::negamax(Board& board, int depth, int alpha, int beta) {
        ++nodes;
        if (should_abort()) { abort_flag = true; return 0; }
    
        uint64_t k = key(board);
        auto it = tt_plain.find(k);
        if (it != tt_plain.end() && it->second.depth >= depth) {
            auto &entry = it->second;
            if (entry.flag == TTFlag::EXACT) return entry.score;
            if (entry.flag == TTFlag::LOWER && entry.score >= beta) return entry.score;
            if (entry.flag == TTFlag::UPPER && entry.score <= alpha) return entry.score;
        }
    
        if (depth == 0 || board.is_win(std::string(1, SYMBOL_X)) || board.is_win(std::string(1, SYMBOL_O)) || board.is_draw()) {
            int val = evaluate_terminal_or_heuristic(board);
            store_tt_plain(k, depth, val, TTFlag::EXACT, std::nullopt);
            return val;
        }
    
        auto moves = board.legal_moves();
        if (moves.empty()) {
            int val = evaluate_terminal_or_heuristic(board);
            store_tt_plain(k, depth, val, TTFlag::EXACT, std::nullopt);
            return val;
        }
    
        std::optional<int> tt_move_local = std::nullopt;
        if (it != tt_plain.end()) tt_move_local = it->second.best_move;
        if (tt_move_local) {
            auto f = std::find(moves.begin(), moves.end(), *tt_move_local);
            if (f != moves.end()) { moves.erase(f); moves.insert(moves.begin(), *tt_move_local); }
        }
    
        int best_score = -10000000;
        std::optional<int> best_move = std::nullopt;
        int original_alpha = alpha;
    
        for (int mv : moves) {
            if (should_abort()) { abort_flag = true; break; }
            try { board.make_move(mv); } catch(...) { continue; }
            int score = -negamax(board, depth - 1, -beta, -alpha);
            try { board.unmake_move(mv); } catch(...) {}
        
            if (score > best_score) { best_score = score; best_move = mv; }
            alpha = std::max(alpha, score);
            if (alpha >= beta) { store_tt_plain(k, depth, best_score, TTFlag::LOWER, best_move); break; }
        }
    
        TTFlag flag;
        if (best_score <= original_alpha) flag = TTFlag::UPPER;
        else if (best_score >= beta) flag = TTFlag::LOWER;
        else flag = TTFlag::EXACT;
    
        store_tt_plain(k, depth, best_score, flag, best_move);
        return best_score;
    }
    
    // ---------- minimax root + recursion ------------------------------------
    
    int Searcher::minimax_root(Board& board, int depth, int alpha, int beta) {
        ++nodes;
        std::string root_player = board.get_side_to_move(); // requires Board accessor
        uint64_t key_plain = key(board);
        uint64_t tk = make_root_key(key_plain, root_player.empty() ? 'X' : root_player[0]);
    
        auto moves = board.legal_moves();
        if (moves.empty()) return evaluate_for_root(board, root_player);
    
        std::optional<int> tt_move = std::nullopt;
        auto itroot = tt_root.find(tk);
        if (itroot != tt_root.end()) tt_move = itroot->second.best_move;
        if (tt_move) {
            auto f = std::find(moves.begin(), moves.end(), *tt_move);
            if (f != moves.end()) { moves.erase(f); moves.insert(moves.begin(), *tt_move); }
        }
    
        std::vector<int> winning, others;
        for (int mv : moves) {
            try { board.make_move(mv); } catch(...) { others.push_back(mv); continue; }
            std::string prev_player = board.get_side_to_move() == std::string(1, SYMBOL_X) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X);
            bool is_win = board.is_win(prev_player);
            try { board.unmake_move(mv); } catch(...) {}
            if (is_win) winning.push_back(mv); else others.push_back(mv);
        }
    
        std::vector<int> ordered;
        ordered.insert(ordered.end(), winning.begin(), winning.end());
        ordered.insert(ordered.end(), others.begin(), others.end());
    
        auto move_score = [&](int mv) -> double {
            double sc = 0.0;
            if (tt_move && *tt_move == mv) sc += 10000000.0;
            auto itkm = killer_moves.find(depth);
            if (itkm != killer_moves.end()) for (int km : itkm->second) if (km == mv) sc += 1000.0;
            auto ith = history.find(mv);
            if (ith != history.end()) sc += static_cast<double>(ith->second);
            sc += -static_cast<double>(mv) * 0.01;
            return sc;
        };
    
        std::sort(ordered.begin(), ordered.end(), [&](int a, int b){ return move_score(a) > move_score(b); });
    
        int best_score = -10000000;
        std::optional<int> best_move = std::nullopt;
        int original_alpha = alpha;
    
        for (int mv : ordered) {
            if (should_abort()) break;
            try { board.make_move(mv); } catch(...) { continue; }
            int score = minimax(board, depth - 1, alpha, beta, root_player);
            try { board.unmake_move(mv); } catch(...) {}
        
            if (score > best_score) { best_score = score; best_move = mv; }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                auto &kms = killer_moves[depth];
                if (std::find(kms.begin(), kms.end(), mv) == kms.end()) { kms.push_back(mv); if (kms.size() > 2) kms.erase(kms.begin()); }
                history[mv] += (1 << depth);
                store_tt_root(tk, depth, best_score, TTFlag::LOWER, best_move);
                break;
            }
        }
    
        TTFlag flag;
        if (best_score <= original_alpha) flag = TTFlag::UPPER;
        else if (best_score >= beta) flag = TTFlag::LOWER;
        else flag = TTFlag::EXACT;
    
        store_tt_root(tk, depth, best_score, flag, best_move);
        try { store_tt_plain(key_plain, depth, best_score, flag, best_move); } catch(...) {}
        return best_score;
    }
    
    int Searcher::minimax(Board& board, int depth, int alpha, int beta, const std::string& root_player) {
        ++nodes;
        if (should_abort()) { abort_flag = true; return 0; }
    
        uint64_t kp = key(board);
        uint64_t tk = make_root_key(kp, root_player.empty() ? 'X' : root_player[0]);
    
        auto it = tt_root.find(tk);
        if (it != tt_root.end() && it->second.depth >= depth) {
            auto &entry = it->second;
            if (entry.flag == TTFlag::EXACT) return entry.score;
            if (entry.flag == TTFlag::LOWER && entry.score >= beta) return entry.score;
            if (entry.flag == TTFlag::UPPER && entry.score <= alpha) return entry.score;
        }
    
        if (depth == 0 || board.is_win(std::string(1, SYMBOL_X)) || board.is_win(std::string(1, SYMBOL_O)) || board.is_draw()) {
            int val = evaluate_for_root(board, root_player);
            store_tt_root(tk, depth, val, TTFlag::EXACT, std::nullopt);
            return val;
        }
    
        auto moves = board.legal_moves();
        if (moves.empty()) {
            int val = evaluate_for_root(board, root_player);
            store_tt_root(tk, depth, val, TTFlag::EXACT, std::nullopt);
            return val;
        }
    
        std::optional<int> tt_move = std::nullopt;
        auto itplain = tt_root.find(tk);
        if (itplain != tt_root.end()) tt_move = itplain->second.best_move;
        if (tt_move) {
            auto f = std::find(moves.begin(), moves.end(), *tt_move);
            if (f != moves.end()) { moves.erase(f); moves.insert(moves.begin(), *tt_move); }
        }
    
        std::vector<int> winning, others;
        for (int mv : moves) {
            try { board.make_move(mv); } catch(...) { others.push_back(mv); continue; }
            std::string prev_player = board.get_side_to_move() == std::string(1, SYMBOL_X) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X);
            bool is_win = board.is_win(prev_player);
            try { board.unmake_move(mv); } catch(...) {}
            if (is_win) winning.push_back(mv); else others.push_back(mv);
        }
    
        std::vector<int> ordered;
        ordered.insert(ordered.end(), winning.begin(), winning.end());
        ordered.insert(ordered.end(), others.begin(), others.end());
    
        auto mv_sort_key = [&](int mv)->double {
            double s = 0.0;
            if (tt_move && *tt_move == mv) s += 10000000.0;
            auto itkm = killer_moves.find(depth);
            if (itkm != killer_moves.end()) for (int km: itkm->second) if (km == mv) s += 1000.0;
            auto ith = history.find(mv);
            if (ith != history.end()) s += static_cast<double>(ith->second);
            s += -static_cast<double>(mv) * 0.01;
            return s;
        };
        std::sort(ordered.begin(), ordered.end(), [&](int a, int b){ return mv_sort_key(a) > mv_sort_key(b); });
    
        bool maximizing = (board.get_side_to_move() == root_player);
        int best_score = maximizing ? -10000000 : 10000000;
        std::optional<int> best_move = std::nullopt;
        int original_alpha = alpha, original_beta = beta;
    
        for (int mv : ordered) {
            if (should_abort()) { abort_flag = true; break; }
            try { board.make_move(mv); } catch(...) { continue; }
            int score = minimax(board, depth - 1, alpha, beta, root_player);
            try { board.unmake_move(mv); } catch(...) {}
        
            if (maximizing) {
                if (score > best_score) { best_score = score; best_move = mv; }
                alpha = std::max(alpha, score);
            } else {
                if (score < best_score) { best_score = score; best_move = mv; }
                beta = std::min(beta, score);
            }
        
            if (alpha >= beta) {
                auto &kms = killer_moves[depth];
                if (std::find(kms.begin(), kms.end(), mv) == kms.end()) { kms.push_back(mv); if (kms.size() > 2) kms.erase(kms.begin()); }
                history[mv] += (1 << depth);
                TTFlag store_flag = maximizing ? TTFlag::LOWER : TTFlag::UPPER;
                store_tt_root(tk, depth, best_score, store_flag, best_move);
                break;
            }
        }
    
        TTFlag final_flag;
        if (maximizing) {
            if (best_score <= original_alpha) final_flag = TTFlag::UPPER;
            else if (best_score >= original_beta) final_flag = TTFlag::LOWER;
            else final_flag = TTFlag::EXACT;
        } else {
            if (best_score >= original_beta) final_flag = TTFlag::LOWER;
            else if (best_score <= original_alpha) final_flag = TTFlag::UPPER;
            else final_flag = TTFlag::EXACT;
        }
    
        store_tt_root(tk, depth, best_score, final_flag, best_move);
        try { store_tt_plain(kp, depth, best_score, final_flag, best_move); } catch(...) {}
        return best_score;
    }
    
    // ---------- PV builders --------------------------------------------------
    
    std::vector<int> Searcher::build_pv_for_root(Board& board, const std::string& root_player) {
        std::vector<int> pv;
        std::vector<int> played;
        try {
            while (true) {
                uint64_t k = key(board);
                uint64_t rk = make_root_key(k, root_player.empty() ? 'X' : root_player[0]);
                auto it = tt_root.find(rk);
                if (it == tt_root.end() || !it->second.best_move.has_value()) break;
                int mv = *it->second.best_move;
                auto legal = board.legal_moves();
                if (std::find(legal.begin(), legal.end(), mv) == legal.end()) break;
                pv.push_back(mv);
                board.make_move(mv);
                played.push_back(mv);
                if (pv.size() > 256) break;
            }
        } catch(...) {}
        for (auto it = played.rbegin(); it != played.rend(); ++it) { try { board.unmake_move(*it); } catch(...) {} }
        return pv;
    }
    
    std::vector<int> Searcher::build_pv(Board& board) {
        std::vector<int> pv;
        try {
            Board cur = board; // copy
            std::unordered_set<uint64_t> seen;
            while (true) {
                uint64_t k = key(cur);
                if (seen.count(k)) break;
                seen.insert(k);
                auto it = tt_plain.find(k);
                if (it == tt_plain.end() || !it->second.best_move.has_value()) break;
                int mv = *it->second.best_move;
                pv.push_back(mv);
                cur.make_move(mv);
            }
        } catch(...) {}
        return pv;
    }
    
    // ---------- Evaluation helpers -------------------------------------------
    
    int Searcher::evaluate_terminal(Board& board) {
        std::string stm = board.get_side_to_move();
        std::string opp = (stm == std::string(1, SYMBOL_X)) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X);
        if (board.is_win(stm)) return SCORE_WIN;
        if (board.is_win(opp)) return SCORE_LOSS;
        if (board.is_draw()) return SCORE_DRAW;
        return 0;
    }
    
    int Searcher::evaluate_terminal_or_heuristic(Board& board) {
        std::string stm = board.get_side_to_move();
        try { return board.evaluate(stm); } catch(...) { return evaluate_terminal(board); }
    }
    
    int Searcher::evaluate_for_root(Board& board, const std::string& root_player) {
        try { return board.evaluate(root_player); } catch(...) {
            if (board.is_win(root_player)) return SCORE_WIN;
            std::string opp = (root_player == std::string(1, SYMBOL_X)) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X);
            if (board.is_win(opp)) return SCORE_LOSS;
            if (board.is_draw()) return SCORE_DRAW;
            return 0;
        }
    }
    
    // ---------- TT helpers ---------------------------------------------------
    
    uint64_t Searcher::key(Board& board) {
        try {
            return board.zobrist_key();
        } catch(...) {
            try {
                std::string s = board.to_string() + "|" + board.get_side_to_move();
                return hash_string_fallback(s);
            } catch(...) {
                std::ostringstream oss; oss << reinterpret_cast<uintptr_t>(&board);
                return hash_string_fallback(oss.str());
            }
        }
    }
    
    void Searcher::store_tt_plain(uint64_t k, int depth, int score, TTFlag flag, std::optional<int> best_move) {
        TTEntry e;
        e.key = k; e.depth = depth; e.score = score; e.flag = flag; e.best_move = best_move;
        tt_plain[k] = std::move(e);
    }
    
    void Searcher::store_tt_root(uint64_t rootk, int depth, int score, TTFlag flag, std::optional<int> best_move) {
        TTEntry e;
        e.key = rootk; e.depth = depth; e.score = score; e.flag = flag; e.best_move = best_move;
        tt_root[rootk] = std::move(e);
    }
    
    // ---------- Limits & abort ----------------------------------------------
    
    bool Searcher::time_exceeded() const {
        if (!time_limit.has_value()) return false;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
        return elapsed >= *time_limit;
    }

    int Searcher::elapsed_ms() const {
        return static_cast<int>(
            duration_cast<milliseconds>(steady_clock::now() - start_time).count()
        );
    }
    
    bool Searcher::nodes_exceeded() const { if (!node_limit.has_value()) return false; return nodes >= *node_limit; }
    
    bool Searcher::should_abort() const { return time_exceeded() || nodes_exceeded(); }
    
    void Searcher::request_abort() { abort_flag = true; }
    
    // ---------- Convenience free function -----------------------------------
    
    Searcher::SearchResult search_position(Board& board,
                                           std::optional<int> max_depth,
                                           std::optional<int> time_ms,
                                           std::optional<int> nodes_limit) {
        Searcher s;
        return s.search(board, max_depth, time_ms, nodes_limit);
    }

} // namespace QuantumOX
