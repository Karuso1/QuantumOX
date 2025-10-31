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
#include "options.h" // used to read "Threads" option dynamically

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <shared_mutex> // for shared mutex

using namespace std::chrono;

namespace QuantumOX {

    // ------------------------ ThreadPool ------------------------------------
    class ThreadPool {
    public:
        ThreadPool(size_t nthreads = std::thread::hardware_concurrency()) : stop_flag(false) {
            if (nthreads == 0) nthreads = 1;
            workers.reserve(nthreads);
            for (size_t i = 0; i < nthreads; ++i) {
                workers.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lk(this->queue_mtx);
                            this->cv.wait(lk, [this] { return this->stop_flag || !this->tasks.empty(); });
                            if (this->stop_flag && this->tasks.empty()) return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        try {
                            task();
                        } catch (...) {
                            // swallow exceptions from tasks to keep pool alive
                        }
                    }
                });
            }
        }

        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lk(queue_mtx);
                stop_flag = true;
            }
            cv.notify_all();
            for (auto &t : workers) if (t.joinable()) t.join();
        }

        // submit a task and get a future
        template<typename F, typename... Args>
        auto submit(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
            using R = typename std::invoke_result_t<F, Args...>;
            auto task_ptr = std::make_shared<std::packaged_task<R()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
            std::future<R> res = task_ptr->get_future();
            {
                std::unique_lock<std::mutex> lk(queue_mtx);
                if (stop_flag) throw std::runtime_error("submit on stopped ThreadPool");
                tasks.emplace([task_ptr]() { (*task_ptr)(); });
            }
            cv.notify_one();
            return res;
        }

        size_t size() const { return workers.size(); }

    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queue_mtx;
        std::condition_variable cv;
        bool stop_flag;
    };

    // ThreadPoolManager: singleton that watches the "Threads" option and
    // recreates the pool if the option value changes.
    class ThreadPoolManager {
    public:
        static ThreadPoolManager& instance() {
            static ThreadPoolManager mgr;
            return mgr;
        }

        // Return a shared_ptr to the current pool; may recreate if option changed.
        std::shared_ptr<ThreadPool> get_pool() {
            std::lock_guard<std::mutex> lk(mtx);
            unsigned int desired = desired_threads_from_option();
            if (!pool || desired != current_count) {
                // recreate
                try {
                    pool = std::make_shared<ThreadPool>(desired);
                    current_count = desired;
                } catch (...) {
                    // fallback to 1
                    pool = std::make_shared<ThreadPool>(1);
                    current_count = 1;
                }
            }
            return pool;
        }

        // Force a refresh (call after options changed externally)
        void refresh() {
            std::lock_guard<std::mutex> lk(mtx);
            unsigned int desired = desired_threads_from_option();
            if (!pool || desired != current_count) {
                try {
                    pool = std::make_shared<ThreadPool>(desired);
                    current_count = desired;
                } catch (...) {
                    pool = std::make_shared<ThreadPool>(1);
                    current_count = 1;
                }
            }
        }

        unsigned int count() const { return current_count; }

    private:
        ThreadPoolManager() {
            current_count = desired_threads_from_option();
            try {
                pool = std::make_shared<ThreadPool>(current_count);
            } catch(...) {
                pool = std::make_shared<ThreadPool>(1);
                current_count = 1;
            }
        }

        // read Threads option; safe if get_option throws -> fallback to hw concurrency or 1
        static unsigned int desired_threads_from_option() {
            try {
                std::string s = get_option("Threads");
                if (!s.empty()) {
                    unsigned long v = std::stoul(s);
                    if (v == 0) v = 1;
                    if (v > 512) v = 512;
                    return static_cast<unsigned int>(v);
                }
            } catch(...) {}
            unsigned int hw = std::thread::hardware_concurrency();
            return hw == 0 ? 1u : hw;
        }

        mutable std::mutex mtx;
        std::shared_ptr<ThreadPool> pool;
        unsigned int current_count{0};
    };

    // ---------------- shared TT (thread-safe) --------------------------------
    // Shared transposition tables across all Searcher instances.
    static std::unordered_map<uint64_t, TTEntry> shared_tt_plain;
    static std::unordered_map<uint64_t, TTEntry> shared_tt_root;
    static std::shared_mutex shared_tt_plain_mtx;
    static std::shared_mutex shared_tt_root_mtx;

    // helpers for shared TT access
    static bool shared_tt_plain_get(uint64_t k, TTEntry &out) {
        std::shared_lock lock(shared_tt_plain_mtx);
        auto it = shared_tt_plain.find(k);
        if (it == shared_tt_plain.end()) return false;
        out = it->second;
        return true;
    }
    static void shared_tt_plain_store(uint64_t k, const TTEntry &e) {
        std::unique_lock lock(shared_tt_plain_mtx);
        shared_tt_plain[k] = e;
    }
    static bool shared_tt_root_get(uint64_t k, TTEntry &out) {
        std::shared_lock lock(shared_tt_root_mtx);
        auto it = shared_tt_root.find(k);
        if (it == shared_tt_root.end()) return false;
        out = it->second;
        return true;
    }
    static void shared_tt_root_store(uint64_t k, const TTEntry &e) {
        std::unique_lock lock(shared_tt_root_mtx);
        shared_tt_root[k] = e;
    }

    // ---------------- small helpers ----------------------------------------
    static uint64_t make_root_key(uint64_t key, char root_char) {
        return key ^ (static_cast<uint64_t>(static_cast<unsigned char>(root_char)) << 56) ^
               0x9e3779b97f4a7c15ULL;
    }
    
    static uint64_t hash_string_fallback(const std::string& s) {
        return static_cast<uint64_t>(std::hash<std::string>{}(s));
    }

    // RAII guard for ply counting (ensures pop on exit)
    struct PlyGuard {
        Searcher* s;
        PlyGuard(Searcher* ss) : s(ss) { s->push_ply(); }
        ~PlyGuard() { s->pop_ply(); }
    }
    ;

    // ---------------- Searcher implementation -------------------------------
    Searcher::Searcher()
        : nodes(0),
          start_time(),
          time_limit(std::nullopt),
          node_limit(std::nullopt),
          abort_flag(false),
          max_seldepth(0),
          current_seldepth(0) {}

    void Searcher::push_ply() {
        ++current_seldepth;
        if (current_seldepth > max_seldepth) max_seldepth = current_seldepth;
    }

    void Searcher::pop_ply() {
        if (current_seldepth > 0) --current_seldepth;
    }

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

    // ---------- helper: elapsed and time checks -----------------------------
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

    // ---------------- core search loop (iterative deepening) ----------------
    Searcher::SearchResult Searcher::search(Board& board,
                                           std::optional<int> max_depth_opt,
                                           std::optional<int> time_ms_opt,
                                           std::optional<int> nodes_limit_opt) {
        int max_depth = max_depth_opt.value_or(DEFAULT_MAX_DEPTH);
                                        
        nodes = 0;
        current_seldepth = 0;
        max_seldepth = 0;
        start_time = std::chrono::steady_clock::now();
        time_limit = time_ms_opt ? std::optional<double>(*time_ms_opt / 1000.0) : std::nullopt;
        node_limit = nodes_limit_opt;
        abort_flag = false;
                                        
        std::optional<int> best_move = std::nullopt;
        int best_score = 0;
        std::vector<int> best_pv;
        std::vector<InfoRecord> infos;

        // get pool from manager (this will respect the Threads option and recreate pool if needed)
        auto pool = ThreadPoolManager::instance().get_pool();

        uint64_t key_plain = key(board);
        std::string root_player = board.get_side_to_move(); // requires accessor in Board
                                        
        const int ASP_WINDOW = 50;
        int prev_neg_score = 0;
        int prev_min_score = 0;
                                        
        for (int depth = 1; depth <= max_depth; ++depth) {
            if (should_abort()) break;

            // reset seldepth counters for this iteration (report per-iteration seldepth)
            max_seldepth = 0;
            current_seldepth = 0;

            // ----------- NEGAMAX pass (root-level parallel per-move) ------------
            int nodes_before = nodes;
            int alpha, beta;
            if (depth > 1) {
                alpha = prev_neg_score - ASP_WINDOW;
                beta = prev_neg_score + ASP_WINDOW;
            } else {
                alpha = -10000000;
                beta = 10000000;
            }

            // we'll call the parallelized helper which uses pool
            int neg_score = negamax_root(board, depth, alpha, beta, depth, pool);

            if (depth > 1 && (neg_score <= alpha || neg_score >= beta)) {
                alpha = -10000000; beta = 10000000;
                neg_score = negamax_root(board, depth, alpha, beta, depth, pool);
            }

            int neg_nodes = nodes - nodes_before;
            prev_neg_score = neg_score;

            if (abort_flag) break;

            // ---------------- negamax PV & move ----------------
            std::optional<int> neg_move = std::nullopt;
            std::vector<int> neg_pv;
            {
                TTEntry e;
                if (shared_tt_plain_get(key_plain, e)) neg_move = e.best_move;
            }
            try { neg_pv = build_pv(board); } 
            catch(...) { if (neg_move) neg_pv = {*neg_move}; }

            if (should_abort()) break;

            // ---------------- MINIMAX pass (root-level parallel per-move) ----------------
            int nodes_before_min = nodes;
            if (depth > 1) {
                alpha = prev_min_score - ASP_WINDOW;
                beta = prev_min_score + ASP_WINDOW;
            } else {
                alpha = -10000000; beta = 10000000;
            }

            int min_score = minimax_root(board, depth, alpha, beta, depth, pool);
            if (depth > 1 && (min_score <= alpha || min_score >= beta)) {
                alpha = -10000000; beta = 10000000;
                min_score = minimax_root(board, depth, alpha, beta, depth, pool);
            }
            int min_nodes = nodes - nodes_before_min;
            prev_min_score = min_score;

            if (abort_flag) break;

            std::optional<int> min_move = std::nullopt;
            std::vector<int> min_pv;
            {
                uint64_t rootk = make_root_key(key_plain, root_player.empty() ? 'X' : root_player[0]);
                TTEntry e;
                if (shared_tt_root_get(rootk, e)) min_move = e.best_move;
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

            // --- record info (also push to infos vector) ---
            InfoRecord ir;
            ir.depth = depth;
            ir.seldepth = max_seldepth;
            ir.score = best_score;
            ir.nodes = nodes;
            ir.time_ms = elapsed_ms();
            ir.negamaxpv = neg_pv;
            ir.minimaxpv = min_pv;
            ir.pv = best_pv;
            infos.push_back(ir);

            std::ostringstream oss;
            oss << "info depth " << depth
                << " seldepth " << max_seldepth
                << " score " << best_score
                << " nodes " << nodes
                << " minimaxpv";
            for (int mv : min_pv) oss << " " << mv;
            oss << " negamaxpv";
            for (int mv : neg_pv) oss << " " << mv;
            oss << " time " << elapsed_ms() << " pv";
            for (int mv : best_pv) oss << " " << mv;
            std::cout << oss.str() << std::endl;

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

    // ---------- negamax root (parallelized per-root-move) --------------------
    // helper result type for tasks
    struct RootEvalResult {
        int move;
        int score;
        std::vector<int> pv; // root move followed by child's pv
        int nodes;
        int seldepth;
    };

    // this variant receives a pool to submit tasks into
    int Searcher::negamax_root(Board& board, int depth, int alpha, int beta, int root_depth, std::shared_ptr<ThreadPool> pool) {
        ++nodes;
        PlyGuard pg(this);

        auto moves = board.legal_moves();
        if (moves.empty()) return evaluate_terminal(board);

        uint64_t k = key(board);
        std::optional<int> tt_move = std::nullopt;
        {
            TTEntry e;
            if (shared_tt_plain_get(k, e)) tt_move = e.best_move;
        }

        if (tt_move) {
            auto f = std::find(moves.begin(), moves.end(), *tt_move);
            if (f != moves.end()) { moves.erase(f); moves.insert(moves.begin(), *tt_move); }
        }

        std::vector<std::future<RootEvalResult>> futures;
        futures.reserve(moves.size());

        // For each root move, submit a task which runs negamax on the child position using a local Searcher
        for (int mv : moves) {
            // capture by value the mv and alpha/beta
            futures.emplace_back(pool->submit([this, board, mv, depth, alpha, beta, root_depth, k]() -> RootEvalResult {
                RootEvalResult rr;
                rr.move = mv;
                rr.nodes = 0;
                rr.seldepth = 0;
                rr.score = -10000000;
                try {
                    Board local_board = board; // copy
                    try { local_board.make_move(mv); } catch(...) { return rr; }

                    // create a local Searcher to avoid sharing per-search state across threads.
                    // tt accesses inside s_local call shared TT helpers via the same member functions.
                    Searcher s_local;
                    s_local.start_time = this->start_time;
                    s_local.time_limit = this->time_limit;
                    s_local.node_limit = this->node_limit;

                    // run the recursive negamax from this child (s_local interacts with shared TT).
                    int child_score = -s_local.negamax(local_board, depth - 1, -beta, -alpha, root_depth);
                    rr.score = child_score;

                    // build pv: root move + child's pv (s_local.build_pv will read shared TT)
                    rr.pv.clear();
                    rr.pv.push_back(mv);
                    try {
                        auto child_pv = s_local.build_pv(local_board);
                        rr.pv.insert(rr.pv.end(), child_pv.begin(), child_pv.end());
                    } catch(...) {}

                    rr.nodes = s_local.nodes;
                    rr.seldepth = s_local.max_seldepth;
                } catch(...) {}
                return rr;
            }));
        }

        int best_score = -10000000;
        std::optional<int> best_move = std::nullopt;

        // collect results
        for (auto &fut : futures) {
            try {
                RootEvalResult rr = fut.get();
                // aggregate nodes and seldepth
                try { this->nodes += rr.nodes; } catch(...) {}
                if (rr.seldepth > this->max_seldepth) this->max_seldepth = rr.seldepth;
                if (rr.score > best_score) {
                    best_score = rr.score;
                    best_move = rr.move;
                    // store PV info into shared TT for this root position (best child)
                    TTEntry e;
                    e.key = k; e.depth = depth; e.score = best_score; e.flag = TTFlag::EXACT; e.best_move = best_move;
                    shared_tt_plain_store(k, e);
                }
            } catch(...) {
                // ignore individual task failure
            }
            if (should_abort()) { abort_flag = true; break; }
        }

        if (!best_move.has_value()) best_score = evaluate_terminal(board);
        return best_score;
    }

    int Searcher::negamax(Board& board, int depth, int alpha, int beta, int root_depth) {
        ++nodes;
        PlyGuard pg(this);

        if (should_abort()) { abort_flag = true; return 0; }

        uint64_t k = key(board);
        TTEntry entry;
        if (shared_tt_plain_get(k, entry) && entry.depth >= depth) {
            if (entry.flag == TTFlag::EXACT) return entry.score;
            if (entry.flag == TTFlag::LOWER && entry.score >= beta) return entry.score;
            if (entry.flag == TTFlag::UPPER && entry.score <= alpha) return entry.score;
        }

        if (depth == 0 || board.is_win(std::string(1, SYMBOL_X)) || board.is_win(std::string(1, SYMBOL_O)) || board.is_draw()) {
            int val = evaluate_terminal_or_heuristic(board);
            TTEntry e; e.key = k; e.depth = depth; e.score = val; e.flag = TTFlag::EXACT; e.best_move = std::nullopt;
            shared_tt_plain_store(k, e);
            return val;
        }

        auto moves = board.legal_moves();
        if (moves.empty()) {
            int val = evaluate_terminal_or_heuristic(board);
            TTEntry e; e.key = k; e.depth = depth; e.score = val; e.flag = TTFlag::EXACT; e.best_move = std::nullopt;
            shared_tt_plain_store(k, e);
            return val;
        }

        std::optional<int> tt_move_local = std::nullopt;
        if (shared_tt_plain_get(k, entry)) tt_move_local = entry.best_move;
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
            int score = -negamax(board, depth - 1, -beta, -alpha, root_depth);
            try { board.unmake_move(mv); } catch(...) {}

            if (score > best_score) { best_score = score; best_move = mv; }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                // Lower bound
                TTEntry e; e.key = k; e.depth = depth; e.score = best_score; e.flag = TTFlag::LOWER; e.best_move = best_move;
                shared_tt_plain_store(k, e);
                break;
            }
        }

        TTFlag flag;
        if (best_score <= original_alpha) flag = TTFlag::UPPER;
        else if (best_score >= beta) flag = TTFlag::LOWER;
        else flag = TTFlag::EXACT;

        TTEntry e; e.key = k; e.depth = depth; e.score = best_score; e.flag = flag; e.best_move = best_move;
        shared_tt_plain_store(k, e);
        return best_score;
    }

    // ---------- minimax root (parallelized per-root-move) --------------------
    int Searcher::minimax_root(Board& board, int depth, int alpha, int beta, int root_depth, std::shared_ptr<ThreadPool> pool) {
        ++nodes;
        PlyGuard pg(this);

        std::string root_player = board.get_side_to_move(); // requires Board accessor
        uint64_t key_plain = key(board);
        uint64_t tk = make_root_key(key_plain, root_player.empty() ? 'X' : root_player[0]);

        auto moves = board.legal_moves();
        if (moves.empty()) return evaluate_for_root(board, root_player);

        std::optional<int> tt_move = std::nullopt;
        {
            TTEntry e;
            if (shared_tt_root_get(tk, e)) tt_move = e.best_move;
        }
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

        std::vector<std::future<RootEvalResult>> futures;
        futures.reserve(ordered.size());

        for (int mv : ordered) {
            futures.emplace_back(pool->submit([this, board, mv, depth, alpha, beta, root_player, root_depth, tk]() -> RootEvalResult {
                RootEvalResult rr;
                rr.move = mv;
                rr.nodes = 0;
                rr.seldepth = 0;
                rr.score = -10000000;
                try {
                    Board local_board = board; // copy
                    try { local_board.make_move(mv); } catch(...) { return rr; }

                    Searcher s_local;
                    s_local.start_time = this->start_time;
                    s_local.time_limit = this->time_limit;
                    s_local.node_limit = this->node_limit;

                    int child_score = s_local.minimax(local_board, depth - 1, alpha, beta, root_player, root_depth);
                    rr.score = child_score;
                    rr.pv.clear();
                    rr.pv.push_back(mv);
                    try {
                        auto child_pv = s_local.build_pv_for_root(local_board, root_player);
                        rr.pv.insert(rr.pv.end(), child_pv.begin(), child_pv.end());
                    } catch(...) {}
                    rr.nodes = s_local.nodes;
                    rr.seldepth = s_local.max_seldepth;
                } catch(...) {}
                return rr;
            }));
        }

        int best_score = -10000000;
        std::optional<int> best_move = std::nullopt;

        for (auto &fut : futures) {
            try {
                RootEvalResult rr = fut.get();
                try { this->nodes += rr.nodes; } catch(...) {}
                if (rr.seldepth > this->max_seldepth) this->max_seldepth = rr.seldepth;
                if (rr.score > best_score) {
                    best_score = rr.score;
                    best_move = rr.move;
                    TTEntry e; e.key = tk; e.depth = depth; e.score = best_score; e.flag = TTFlag::EXACT; e.best_move = best_move;
                    shared_tt_root_store(tk, e);
                }
            } catch(...) {}
            if (should_abort()) { abort_flag = true; break; }
        }

        if (!best_move.has_value()) best_score = evaluate_for_root(board, root_player);
        return best_score;
    }

    int Searcher::minimax(Board& board, int depth, int alpha, int beta, const std::string& root_player, int root_depth) {
        ++nodes;
        PlyGuard pg(this);

        if (should_abort()) { abort_flag = true; return 0; }

        uint64_t kp = key(board);
        uint64_t tk = make_root_key(kp, root_player.empty() ? 'X' : root_player[0]);

        TTEntry entry;
        if (shared_tt_root_get(tk, entry) && entry.depth >= depth) {
            if (entry.flag == TTFlag::EXACT) return entry.score;
            if (entry.flag == TTFlag::LOWER && entry.score >= beta) return entry.score;
            if (entry.flag == TTFlag::UPPER && entry.score <= alpha) return entry.score;
        }

        if (depth == 0 || board.is_win(std::string(1, SYMBOL_X)) || board.is_win(std::string(1, SYMBOL_O)) || board.is_draw()) {
            int val = evaluate_for_root(board, root_player);
            TTEntry e; e.key = tk; e.depth = depth; e.score = val; e.flag = TTFlag::EXACT; e.best_move = std::nullopt;
            shared_tt_root_store(tk, e);
            return val;
        }

        auto moves = board.legal_moves();
        if (moves.empty()) {
            int val = evaluate_for_root(board, root_player);
            TTEntry e; e.key = tk; e.depth = depth; e.score = val; e.flag = TTFlag::EXACT; e.best_move = std::nullopt;
            shared_tt_root_store(tk, e);
            return val;
        }

        std::optional<int> tt_move = std::nullopt;
        if (shared_tt_root_get(tk, entry)) tt_move = entry.best_move;
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
            int score = minimax(board, depth - 1, alpha, beta, root_player, root_depth);
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
                TTEntry e; e.key = tk; e.depth = depth; e.score = best_score; e.flag = store_flag; e.best_move = best_move;
                shared_tt_root_store(tk, e);
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

        TTEntry e; e.key = tk; e.depth = depth; e.score = best_score; e.flag = final_flag; e.best_move = best_move;
        shared_tt_root_store(tk, e);
        // Also attempt to store plain TT for this kp (best effort)
        try { 
            TTEntry pe; pe.key = kp; pe.depth = depth; pe.score = best_score; pe.flag = final_flag; pe.best_move = best_move;
            shared_tt_plain_store(kp, pe);
        } catch(...) {}
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
                TTEntry e;
                if (!shared_tt_root_get(rk, e) || !e.best_move.has_value()) break;
                int mv = *e.best_move;
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
                TTEntry e;
                if (!shared_tt_plain_get(k, e) || !e.best_move.has_value()) break;
                int mv = *e.best_move;
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

    // ---------- TT helpers (instance methods now forward to shared TT) -------
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
        shared_tt_plain_store(k, e);
    }

    void Searcher::store_tt_root(uint64_t rootk, int depth, int score, TTFlag flag, std::optional<int> best_move) {
        TTEntry e;
        e.key = rootk; e.depth = depth; e.score = score; e.flag = flag; e.best_move = best_move;
        shared_tt_root_store(rootk, e);
    }

    // ---------- Convenience free function -----------------------------------

    Searcher::SearchResult search_position(Board& board,
                                           std::optional<int> max_depth,
                                           std::optional<int> time_ms,
                                           std::optional<int> nodes_limit) {
        Searcher s;
        return s.search(board, max_depth, time_ms, nodes_limit);
    }

} // namespace QuantumOX
