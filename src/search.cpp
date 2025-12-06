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
#include <cmath>
#include <numeric> // for accumulate

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
        std::shared_lock<std::shared_mutex> lock(shared_tt_plain_mtx);
        auto it = shared_tt_plain.find(k);
        if (it == shared_tt_plain.end()) return false;
        out = it->second;
        return true;
    }
    static void shared_tt_plain_store(uint64_t k, const TTEntry &e) {
        std::unique_lock<std::shared_mutex> lock(shared_tt_plain_mtx);
        shared_tt_plain[k] = e;
    }
    static bool shared_tt_root_get(uint64_t k, TTEntry &out) {
        std::shared_lock<std::shared_mutex> lock(shared_tt_root_mtx);
        auto it = shared_tt_root.find(k);
        if (it == shared_tt_root.end()) return false;
        out = it->second;
        return true;
    }
    static void shared_tt_root_store(uint64_t k, const TTEntry &e) {
        std::unique_lock<std::shared_mutex> lock(shared_tt_root_mtx);
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

    // compute split depth: given an integer depth d, compute d/2 and apply rule:
    // if fractional part < 0.5 -> floor, if >= 0.5 -> ceil.
    static int split_depth_for(int depth) {
        double half = static_cast<double>(depth) / 2.0;
        double fl = std::floor(half);
        double frac = half - fl;
        int base = static_cast<int>(fl);
        if (frac >= 0.5) return base + 1;
        return base;
    }

    // RAII guard for ply counting (ensures pop on exit)
    struct PlyGuard {
        Searcher* s;
        PlyGuard(Searcher* ss) : s(ss) { s->push_ply(); }
        ~PlyGuard() { s->pop_ply(); }
    }
    ;

    // ---------------- parameters for heuristics (tweakable) ------------------
    namespace {
        constexpr int INF = 10000000;
        constexpr int ASP_WINDOW = 50;
        constexpr int LMR_DEPTH_THRESHOLD = 2; // only reduce when depth > this
        // base reduction formula: reduce = 1 + log(depth) approx
        inline int lmr_reduction(int depth, int move_index) {
            // more reduction for later moves and deeper depth
            if (depth <= LMR_DEPTH_THRESHOLD) return 0;
            double d = std::log2(static_cast<double>(depth));
            int r = 1 + static_cast<int>(d * (0.8 + move_index / 8.0));
            // clamp
            return std::min(r, depth - 1);
        }
    }

    // ---------------- Searcher implementation -------------------------------
    Searcher::Searcher()
        : nodes(0),
          start_time(),
          time_limit(std::nullopt),
          node_limit(std::nullopt),
          abort_flag(std::make_shared<std::atomic<bool>>(false)),
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
        int val = 0;
        try {
            val = board.evaluate(for_player);
        } catch (...) {
            val = 0;
        }
        try { board.unmake_move(mv); } catch(...) {}
        return val;
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

    bool Searcher::should_abort() const {
        bool ab = abort_flag ? abort_flag->load() : false;
        return ab || time_exceeded() || nodes_exceeded();
    }

    void Searcher::request_abort() {
        if (abort_flag) abort_flag->store(true);
    }

    // ---------- Utility: order moves with advanced heuristics ---------------
    // returns vector of moves sorted (best first) but DOES NOT modify board.
    std::vector<int> order_moves_for_negamax(Searcher* s, Board& board, std::vector<int> moves, uint64_t k, int depth) {
        struct MoveKey { int mv; double key; size_t orig; };
        std::vector<MoveKey> mk;
        mk.reserve(moves.size());

        // read TT move and killer/history
        std::optional<int> tt_move = std::nullopt;
        {
            TTEntry e;
            if (shared_tt_plain_get(k, e)) tt_move = e.best_move;
        }
        for (size_t i = 0; i < moves.size(); ++i) {
            int mv = moves[i];
            double score = 0.0;
            if (tt_move && *tt_move == mv) score += 1e8;
            auto itkm = s->killer_moves.find(depth);
            if (itkm != s->killer_moves.end()) {
                for (int km : itkm->second) if (km == mv) score += 5000.0;
            }
            auto ith = s->history.find(mv);
            if (ith != s->history.end()) score += static_cast<double>(ith->second);
            // quick heuristic: immediate winning move is huge priority
            try {
                board.make_move(mv);
                std::string prev = board.get_side_to_move() == std::string(1, SYMBOL_X) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X);
                if (board.is_win(prev)) score += 1e6;
                board.unmake_move(mv);
            } catch(...) {}
            // small bias to stabilize order
            score += -static_cast<double>(mv) * 0.01;
            mk.push_back({mv, score, i});
        }
        std::sort(mk.begin(), mk.end(), [](const MoveKey& a, const MoveKey& b){ return a.key > b.key; });
        std::vector<int> out; out.reserve(mk.size());
        for (auto &m : mk) out.push_back(m.mv);
        return out;
    }

    std::vector<int> order_moves_for_minimax(Searcher* s, Board& board, std::vector<int> moves, uint64_t tk, int depth) {
        struct MoveKey { int mv; double key; size_t orig; };
        std::vector<MoveKey> mk;
        mk.reserve(moves.size());

        std::optional<int> tt_move = std::nullopt;
        {
            TTEntry e;
            if (shared_tt_root_get(tk, e)) tt_move = e.best_move;
        }

        for (size_t i = 0; i < moves.size(); ++i) {
            int mv = moves[i];
            double score = 0.0;
            if (tt_move && *tt_move == mv) score += 1e8;
            auto itkm = s->killer_moves.find(depth);
            if (itkm != s->killer_moves.end()) {
                for (int km : itkm->second) if (km == mv) score += 5000.0;
            }
            auto ith = s->history.find(mv);
            if (ith != s->history.end()) score += static_cast<double>(ith->second);
            try {
                board.make_move(mv);
                std::string prev = board.get_side_to_move() == std::string(1, SYMBOL_X) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X);
                if (board.is_win(prev)) score += 1e6;
                board.unmake_move(mv);
            } catch(...) {}
            score += -static_cast<double>(mv) * 0.01;
            mk.push_back({mv, score, i});
        }
        std::sort(mk.begin(), mk.end(), [](const MoveKey& a, const MoveKey& b){ return a.key > b.key; });
        std::vector<int> out; out.reserve(mk.size());
        for (auto &m : mk) out.push_back(m.mv);
        return out;
    }

    auto info_token = [](const std::string &token) {
        std::cout << token << " " << std::flush; // flush makes it appear immediately
    };

    // ---------- RootEvalResult (moved up so it's usable during logging) ------
    struct RootEvalResult {
        int move;
        int score;
        std::vector<int> pv; // root move followed by child's pv
        int nodes;
        int seldepth;
    };

    // ---------------- core search loop (iterative deepening) ----------------
    // We'll capture seldepth for negamax and minimax separately, then combine them
    // (sum) and print that as seldepth in info lines. We don't change search logic,
    // only track and aggregate seldepth values.
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
        if (!abort_flag) abort_flag = std::make_shared<std::atomic<bool>>(false);
        else abort_flag->store(false);

        std::optional<int> best_move = std::nullopt;
        int best_score = 0;
        std::vector<int> best_pv;
        std::vector<InfoRecord> infos;

        auto pool = ThreadPoolManager::instance().get_pool();
        unsigned int threads = ThreadPoolManager::instance().count();
        std::cout << "info string Using " << threads << " thread" << (threads == 1 ? "" : "s") << std::endl;

        uint64_t key_plain = key(board);
        std::string root_player = board.get_side_to_move();

        int prev_neg_score = 0;
        int prev_min_score = 0;

        uint64_t hash_size_entries = 0;
        try {
            hash_size_entries = std::stoull(get_option("Hash"));
            if (hash_size_entries == 0) hash_size_entries = 1;
        } catch (...) { hash_size_entries = 16; } // fallback

        for (int depth = 1; depth <= max_depth; ++depth) {
            if (should_abort()) break;

            // reset per-iteration seldepth trackers
            int neg_seldepth = 0;
            int min_seldepth = 0;
            int combined_seldepth = 0;

            // keep global max_seldepth for internal bookkeeping
            max_seldepth = 0;
            current_seldepth = 0;

            // compute split depth
            int per_algo_depth = split_depth_for(depth);
            // ensure non-negative
            if (per_algo_depth < 0) per_algo_depth = 0;

            // Negamax aspiration window based on prev_neg_score
            int alpha_neg, beta_neg;
            if (depth > 1) {
                alpha_neg = prev_neg_score - ASP_WINDOW;
                beta_neg  = prev_neg_score + ASP_WINDOW;
            } else {
                alpha_neg = -INF; beta_neg = INF;
            }

            // Minimax aspiration window based on prev_min_score (we prepare both windows so we can run in parallel)
            int alpha_min, beta_min;
            if (depth > 1) {
                alpha_min = prev_min_score - ASP_WINDOW;
                beta_min  = prev_min_score + ASP_WINDOW;
            } else {
                alpha_min = -INF; beta_min = INF;
            }

            int neg_score = 0;
            int min_score = 0;
            std::vector<int> neg_pv, min_pv;
            std::optional<int> neg_move = std::nullopt, min_move = std::nullopt;

            // Helper struct for returning algorithm results from async/sequential runners
            struct AlgoResult { int score; int seldepth; int nodes; std::vector<RootEvalResult> root_moves; };

            // containers to collect per-root-move results when we invoke roots (used for hybrid logging)
            std::vector<RootEvalResult> neg_root_results;
            std::vector<RootEvalResult> min_root_results;

            // If exactly 2 threads are configured, run minimax & negamax concurrently,
            // each using a tiny local pool of 1 thread to avoid excessive nested parallelism.
            if (threads == 2) {
                auto local_pool_neg = std::make_shared<ThreadPool>(1);
                auto local_pool_min = std::make_shared<ThreadPool>(1);

                // Launch negamax on separate Searcher runner (so we can capture its max_seldepth)
                auto neg_fut = std::async(std::launch::async, [this, b = board, per_algo_depth, alpha_neg, beta_neg, depth, local_pool_neg]() mutable -> AlgoResult {
                    AlgoResult ar; ar.score = -INF; ar.seldepth = 0; ar.nodes = 0;
                    try {
                        Searcher runner;
                        runner.abort_flag = this->abort_flag;
                        runner.start_time = this->start_time;
                        runner.time_limit = this->time_limit;
                        runner.node_limit = this->node_limit;
                        // capture move-by-move results into local vector
                        std::vector<RootEvalResult> local_moves;
                        int sc = runner.negamax_root(const_cast<Board&>(b), per_algo_depth, alpha_neg, beta_neg, depth, local_pool_neg,
                            // on_move callback
                            [&](const RootEvalResult& rr) {
                                local_moves.push_back(rr);
                            }
                        );
                        ar.score = sc;
                        ar.seldepth = runner.max_seldepth;
                        ar.nodes = runner.nodes;
                        ar.root_moves = std::move(local_moves);
                    } catch(...) {}
                    return ar;
                });

                // Launch minimax on separate Searcher runner
                auto min_fut = std::async(std::launch::async, [this, b = board, per_algo_depth, alpha_min, beta_min, depth, local_pool_min]() mutable -> AlgoResult {
                    AlgoResult ar; ar.score = -INF; ar.seldepth = 0; ar.nodes = 0;
                    try {
                        auto pool_for_min = std::make_shared<ThreadPool>(1);
                        Searcher runner;
                        runner.abort_flag = this->abort_flag;
                        runner.start_time = this->start_time;
                        runner.time_limit = this->time_limit;
                        runner.node_limit = this->node_limit;
                        std::vector<RootEvalResult> local_moves;
                        int sc = runner.minimax_root(const_cast<Board&>(b), per_algo_depth, alpha_min, beta_min, depth, pool_for_min,
                            [&](const RootEvalResult& rr) {
                                local_moves.push_back(rr);
                            }
                        );
                        ar.score = sc;
                        ar.seldepth = runner.max_seldepth;
                        ar.nodes = runner.nodes;
                        ar.root_moves = std::move(local_moves);
                    } catch(...) {}
                    return ar;
                });

                // collect results (and handle aspiration fail re-searchs sequentially if needed)
                AlgoResult neg_res; neg_res.score = -INF; neg_res.seldepth = 0; neg_res.nodes = 0;
                AlgoResult min_res; min_res.score = -INF; min_res.seldepth = 0; min_res.nodes = 0;
                try { neg_res = neg_fut.get(); } catch(...) { neg_res.score = -INF; }
                // aspiration fail -> re-search negamax full window if needed
                if (per_algo_depth > 0 && (depth > 1) && (neg_res.score <= alpha_neg || neg_res.score >= beta_neg)) {
                    try {
                        Searcher runner;
                        runner.abort_flag = this->abort_flag;
                        runner.start_time = this->start_time;
                        runner.time_limit = this->time_limit;
                        runner.node_limit = this->node_limit;
                        std::vector<RootEvalResult> extra_moves;
                        auto pool_re = std::make_shared<ThreadPool>(1);
                        int sc = runner.negamax_root(board, per_algo_depth, -INF, INF, depth, pool_re,
                            [&](const RootEvalResult& rr) { extra_moves.push_back(rr); }
                        );
                        neg_res.score = sc;
                        neg_res.seldepth = std::max(neg_res.seldepth, runner.max_seldepth);
                        neg_res.nodes += runner.nodes;
                        // append any extra move records (best-effort)
                        neg_res.root_moves.insert(neg_res.root_moves.end(), extra_moves.begin(), extra_moves.end());
                    } catch(...) {}
                }

                try { min_res = min_fut.get(); } catch(...) { min_res.score = -INF; }
                // aspiration fail -> re-search minimax full window if needed
                if (per_algo_depth > 0 && (depth > 1) && (min_res.score <= alpha_min || min_res.score >= beta_min)) {
                    try {
                        Searcher runner;
                        runner.abort_flag = this->abort_flag;
                        runner.start_time = this->start_time;
                        runner.time_limit = this->time_limit;
                        runner.node_limit = this->node_limit;
                        std::vector<RootEvalResult> extra_moves2;
                        auto pool_re2 = std::make_shared<ThreadPool>(1);
                        int sc = runner.minimax_root(board, per_algo_depth, -INF, INF, depth, pool_re2,
                            [&](const RootEvalResult& rr) { extra_moves2.push_back(rr); }
                        );
                        min_res.score = sc;
                        min_res.seldepth = std::max(min_res.seldepth, runner.max_seldepth);
                        min_res.nodes += runner.nodes;
                        min_res.root_moves.insert(min_res.root_moves.end(), extra_moves2.begin(), extra_moves2.end());
                    } catch(...) {}
                }

                // add nodes from runs into this Searcher's nodes
                try { this->nodes += neg_res.nodes; } catch(...) {}
                try { this->nodes += min_res.nodes; } catch(...) {}

                // update per-algo seldepths and global max_seldepth (for internal tracking)
                neg_seldepth = neg_res.seldepth;
                min_seldepth = min_res.seldepth;
                if (neg_seldepth > this->max_seldepth) this->max_seldepth = neg_seldepth;
                if (min_seldepth > this->max_seldepth) this->max_seldepth = min_seldepth;

                neg_score = neg_res.score;
                min_score = min_res.score;

                // After both roots run, extract PVs/moves from TT like before
                {
                    TTEntry e;
                    if (shared_tt_plain_get(key_plain, e)) neg_move = e.best_move;
                }
                try { neg_pv = build_pv(board); } catch(...) { if (neg_move) neg_pv = {*neg_move}; }

                {
                    uint64_t rootk = make_root_key(key_plain, root_player.empty() ? 'X' : root_player[0]);
                    TTEntry e;
                    if (shared_tt_root_get(rootk, e)) min_move = e.best_move;
                }
                try { min_pv = build_pv_for_root(board, root_player); }
                catch(...) {
                    try { min_pv = build_pv(board); } catch(...) { if (min_move) min_pv = {*min_move}; }
                }

                // move results back to local collectors for hybrid printing
                neg_root_results = std::move(neg_res.root_moves);
                min_root_results = std::move(min_res.root_moves);

            } else {
                // default behavior (sequential) but still split depth for each algorithm as requested:
                // run negamax (with per_algo_depth) on a runner so we can capture seldepth
                {
                    Searcher runner;
                    runner.abort_flag = this->abort_flag;
                    runner.start_time = this->start_time;
                    runner.time_limit = this->time_limit;
                    runner.node_limit = this->node_limit;
                    std::vector<RootEvalResult> local_moves;
                    try {
                        int sc = runner.negamax_root(board, per_algo_depth, alpha_neg, beta_neg, depth, pool,
                            [&](const RootEvalResult& rr) { local_moves.push_back(rr); }
                        );
                        neg_score = sc;
                        neg_root_results = std::move(local_moves);
                    } catch(...) { neg_score = -INF; }
                    // collect runner stats
                    try { this->nodes += runner.nodes; } catch(...) {}
                    neg_seldepth = runner.max_seldepth;
                    if (neg_seldepth > this->max_seldepth) this->max_seldepth = neg_seldepth;
                }

                // aspiration fail -> full-window re-search
                if (per_algo_depth > 0 && (depth > 1) && (neg_score <= alpha_neg || neg_score >= beta_neg)) {
                    Searcher runner2;
                    runner2.abort_flag = this->abort_flag;
                    runner2.start_time = this->start_time;
                    runner2.time_limit = this->time_limit;
                    runner2.node_limit = this->node_limit;
                    std::vector<RootEvalResult> local_moves2;
                    try {
                        int sc = runner2.negamax_root(board, per_algo_depth, -INF, INF, depth, pool,
                            [&](const RootEvalResult& rr) { local_moves2.push_back(rr); }
                        );
                        neg_score = sc;
                        // append results
                        for (auto &r : local_moves2) neg_root_results.push_back(r);
                    } catch(...) { neg_score = -INF; }
                    try { this->nodes += runner2.nodes; } catch(...) {}
                    neg_seldepth = std::max(neg_seldepth, runner2.max_seldepth);
                    if (neg_seldepth > this->max_seldepth) this->max_seldepth = neg_seldepth;
                }

                {
                    TTEntry e;
                    if (shared_tt_plain_get(key_plain, e)) neg_move = e.best_move;
                }
                try { neg_pv = build_pv(board); } catch(...) { if (neg_move) neg_pv = {*neg_move}; }

                if (should_abort()) break;

                // Minimax pass (sequential), also with per_algo_depth on a runner
                {
                    Searcher runner_m;
                    runner_m.abort_flag = this->abort_flag;
                    runner_m.start_time = this->start_time;
                    runner_m.time_limit = this->time_limit;
                    runner_m.node_limit = this->node_limit;
                    std::vector<RootEvalResult> local_moves_m;
                    try {
                        int sc = runner_m.minimax_root(board, per_algo_depth, alpha_min, beta_min, depth, pool,
                            [&](const RootEvalResult& rr) { local_moves_m.push_back(rr); }
                        );
                        min_score = sc;
                        min_root_results = std::move(local_moves_m);
                    } catch(...) { min_score = -INF; }
                    try { this->nodes += runner_m.nodes; } catch(...) {}
                    min_seldepth = runner_m.max_seldepth;
                    if (min_seldepth > this->max_seldepth) this->max_seldepth = min_seldepth;
                }

                if (per_algo_depth > 0 && (depth > 1) && (min_score <= alpha_min || min_score >= beta_min)) {
                    Searcher runner_m2;
                    runner_m2.abort_flag = this->abort_flag;
                    runner_m2.start_time = this->start_time;
                    runner_m2.time_limit = this->time_limit;
                    runner_m2.node_limit = this->node_limit;
                    std::vector<RootEvalResult> local_moves_m2;
                    try {
                        int sc = runner_m2.minimax_root(board, per_algo_depth, -INF, INF, depth, pool,
                            [&](const RootEvalResult& rr) { local_moves_m2.push_back(rr); }
                        );
                        min_score = sc;
                        for (auto &r : local_moves_m2) min_root_results.push_back(r);
                    } catch(...) { min_score = -INF; }
                    try { this->nodes += runner_m2.nodes; } catch(...) {}
                    min_seldepth = std::max(min_seldepth, runner_m2.max_seldepth);
                    if (min_seldepth > this->max_seldepth) this->max_seldepth = min_seldepth;
                }

                {
                    uint64_t rootk = make_root_key(key_plain, root_player.empty() ? 'X' : root_player[0]);
                    TTEntry e;
                    if (shared_tt_root_get(rootk, e)) min_move = e.best_move;
                }
                try { min_pv = build_pv_for_root(board, root_player); }
                catch(...) {
                    try { min_pv = build_pv(board); } catch(...) { if (min_move) min_pv = {*min_move}; }
                }
            }

            // combine seldepths (we sum both algorithms' seldepth values)
            combined_seldepth = neg_seldepth + min_seldepth;
            // also keep global max_seldepth to reflect maximum reached by any subsearch (for backward compatibility)
            if (combined_seldepth < this->max_seldepth) {
                // preserve the internal max if it is larger than sum (rare), else we keep sum
                combined_seldepth = std::max(combined_seldepth, this->max_seldepth);
            }

            prev_neg_score = neg_score;
            prev_min_score = min_score;

            if (should_abort()) break;

            // decide which algorithm to prefer for chosen move (this chooses best overall PV to use)
            std::string selector = "negamax";
            int chosen_score = neg_score;
            std::optional<int> chosen_move = neg_move;
            std::vector<int> chosen_pv = neg_pv;

            if (min_score > neg_score) {
                selector = "minimax";
                chosen_score = min_score;
                chosen_move = min_move;
                chosen_pv = min_pv;
            } else if (min_score == neg_score) {
                // prefer the one with smaller node count to reduce noise
                // (we didn't calculate node counts separately here; keep negamax preference unless minimax stored better move)
            }

            if (chosen_move.has_value()) {
                best_move = chosen_move;
                best_score = chosen_score;
                best_pv = chosen_pv;
            }

            // record info (use combined_seldepth for seldepth)
            InfoRecord ir;
            ir.depth = depth;
            ir.seldepth = combined_seldepth;
            ir.score = best_score;
            ir.nodes = nodes;
            ir.time_ms = elapsed_ms();
            ir.nps = (ir.time_ms > 0) ? (nodes * 1000LL / ir.time_ms) : 0;
            ir.negamaxpv = neg_pv;
            ir.minimaxpv = min_pv;
            ir.pv = best_pv;
            infos.push_back(ir);

            // compute hashfull per-mille
            uint64_t current_tt_entries = shared_tt_plain.size(); // plain TT
            uint64_t hashfull_permille = static_cast<uint64_t>(
                std::min(1000.0, (double(current_tt_entries) * 1000.0 / double(hash_size_entries)))
            );

            // --- Special hybrid logging for depths >= 34 ---
            bool hybrid = (depth >= 34);

            if (!hybrid) {
                info_token("info");
                info_token("depth"); info_token(std::to_string(depth));
                info_token("seldepth"); info_token(std::to_string(ir.seldepth));
                info_token("score"); info_token(std::to_string(best_score));
                info_token("nodes"); info_token(std::to_string(nodes));
                info_token("nps"); info_token(std::to_string(ir.nps));
                info_token("hashfull"); info_token(std::to_string(hashfull_permille));
                info_token("time"); info_token(std::to_string(elapsed_ms()));
                info_token("pv");
                for (int mv : best_pv) info_token(std::to_string(mv));

                // finally end line
                std::cout << std::endl;
            } else {
                // ----- HYBRID MODE LOGIC START -----
                // We have per-root move lists from both algorithms in:
                //   neg_root_results (order is the order the negamax root evaluated moves)
                //   min_root_results (order is the order the minimax root evaluated moves)
                //
                // We'll print currmove lines for each currmovenumber.
                // The "currmovenumber" is defined per the algorithm's generation ordering.
                //
                // Determine starting algorithm for alternation:
                std::string start_alg = (min_score >= neg_score) ? "minimax" : "minimax"; // prefer minimax as typical default per spec
                // (spec mostly uses minimax as starting default; but if you want to change to negamax if it was clearly higher, you can)
                // For deterministic behavior: follow spec and start with minimax.

                // Prepare maps for quick index lookup
                // root_results vectors contain RootEvalResult in the order each algorithm generated them.
                size_t max_moves = std::max(neg_root_results.size(), min_root_results.size());
                if (max_moves == 0) max_moves = 0; // safe guard

                // vectors to accumulate "selected most-score moves" per algorithm to compute overall
                std::vector<double> selected_scores_minimax;
                std::vector<double> selected_scores_negamax;

                // alternate pattern per currmovenumber. Pattern of tests: start_alg, other, start, other (up to 4 tests),
                // printing each test line. After tests, choose default for next moves based on highest score seen in tests
                for (size_t idx = 0; idx < max_moves; ++idx) {
                    size_t currmovenumber = idx + 1;
                    // test order: 0..3 -> alternate starting with start_alg
                    for (int t = 0; t < 4; ++t) {
                        bool use_start_alg = (t % 2 == 0);
                        std::string alg = use_start_alg ? start_alg : (start_alg == "minimax" ? "negamax" : "minimax");
                        // determine which algorithm's move at this index exists
                        if (alg == "minimax") {
                            if (idx < min_root_results.size()) {
                                auto &rr = min_root_results[idx];
                                // print test line
                                std::cout << "info depth " << depth << " currmove " << rr.move << " currmovenumber " << currmovenumber << " algorithm minimax" << std::endl;
                            } else {
                                // no move for this index in minimax, skip print
                            }
                        } else { // negamax
                            if (idx < neg_root_results.size()) {
                                auto &rr = neg_root_results[idx];
                                std::cout << "info depth " << depth << " currmove " << rr.move << " currmovenumber " << currmovenumber << " algorithm negamax" << std::endl;
                            } else {
                                // no move for this index in negamax, skip
                            }
                        }
                        // allow abort if needed
                        if (should_abort()) break;
                    }
                    if (should_abort()) break;

                    // After the test cycle for this currmovenumber choose default algorithm to generate next moves.
                    // Decision rule:
                    //  - Compare the last available score of minimax and negamax for this index (if absent treat as -INF)
                    int mm_score = (idx < min_root_results.size()) ? min_root_results[idx].score : -INF;
                    int nn_score = (idx < neg_root_results.size()) ? neg_root_results[idx].score : -INF;

                    std::string chosen_alg_for_index = "minimax";
                    if (nn_score > mm_score) chosen_alg_for_index = "negamax";
                    else if (mm_score > nn_score) chosen_alg_for_index = "minimax";
                    else chosen_alg_for_index = "minimax"; // tie -> prefer minimax

                    if (chosen_alg_for_index == "minimax") {
                        if (idx < min_root_results.size()) selected_scores_minimax.push_back(static_cast<double>(min_root_results[idx].score));
                    } else {
                        if (idx < neg_root_results.size()) selected_scores_negamax.push_back(static_cast<double>(neg_root_results[idx].score));
                    }

                    // compute overall for chosen algorithm
                    double overall = 0.0;
                    if (chosen_alg_for_index == "minimax") {
                        if (!selected_scores_minimax.empty()) {
                            overall = std::accumulate(selected_scores_minimax.begin(), selected_scores_minimax.end(), 0.0) / selected_scores_minimax.size();
                        } else overall = 0.0;
                    } else {
                        if (!selected_scores_negamax.empty()) {
                            overall = std::accumulate(selected_scores_negamax.begin(), selected_scores_negamax.end(), 0.0) / selected_scores_negamax.size();
                        } else overall = 0.0;
                    }

                    // round overall normally
                    long overall_rounded = static_cast<long>(std::floor(overall + 0.5));

                    // Print final default line for this currmovenumber
                    std::cout << "info depth " << depth << " currmove ";
                    if (chosen_alg_for_index == "minimax") {
                        // print chosen move (if exists)
                        if (idx < min_root_results.size()) std::cout << min_root_results[idx].move;
                        else if (idx < neg_root_results.size()) std::cout << neg_root_results[idx].move;
                        else std::cout << 0;
                        std::cout << " currmovenumber " << currmovenumber << " algorithm default to minimax (overall " << overall_rounded << ")" << std::endl;
                    } else {
                        if (idx < neg_root_results.size()) std::cout << neg_root_results[idx].move;
                        else if (idx < min_root_results.size()) std::cout << min_root_results[idx].move;
                        else std::cout << 0;
                        std::cout << " currmovenumber " << currmovenumber << " algorithm default to negamax (overall " << overall_rounded << ")" << std::endl;
                    }

                    // The "default" chosen for this currmovenumber could influence the chosen PV at the end of the depth.
                    // But we will still use the previously chosen overall PV (chosen_pv), which was selected from per-algo root scores.
                    // Next currmovenumber continues...
                } // end for each currmovenumber

                // After printing all currmove sequences, now print the "normal final info line" for the depth
                info_token("info");
                info_token("depth"); info_token(std::to_string(depth));
                info_token("seldepth"); info_token(std::to_string(ir.seldepth));
                info_token("score"); info_token(std::to_string(best_score));
                info_token("nodes"); info_token(std::to_string(nodes));
                info_token("nps"); info_token(std::to_string(ir.nps));
                info_token("hashfull"); info_token(std::to_string(hashfull_permille));
                info_token("time"); info_token(std::to_string(elapsed_ms()));
                info_token("pv");
                for (int mv : chosen_pv) info_token(std::to_string(mv));
                std::cout << std::endl;
                // ----- HYBRID MODE LOGIC END -----
            }

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
    // Note: added optional callback parameter `on_move` so callers (search()) can
    // receive per-root-move evaluation records for hybrid logging. Default null -> no callback.
    int Searcher::negamax_root(Board& board, int depth, int alpha, int beta, int root_depth, std::shared_ptr<ThreadPool> pool, std::function<void(const RootEvalResult&)> on_move) {
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

        // advanced ordering (local)
        std::vector<int> ordered = order_moves_for_negamax(this, board, moves, k, depth);

        std::vector<std::future<RootEvalResult>> futures;
        futures.reserve(ordered.size());

        for (int mv : ordered) {
            futures.emplace_back(pool->submit([this, board, mv, depth, alpha, beta, root_depth, k]() -> RootEvalResult {
                RootEvalResult rr;
                rr.move = mv;
                rr.nodes = 0;
                rr.seldepth = 0;
                rr.score = -INF;
                try {
                    Board local_board = board; // copy
                    try { local_board.make_move(mv); } catch(...) { return rr; }

                    Searcher s_local;
                    s_local.abort_flag = this->abort_flag;
                    s_local.start_time = this->start_time;
                    s_local.time_limit = this->time_limit;
                    s_local.node_limit = this->node_limit;

                    // principal variation search style:
                    int child_score = -s_local.negamax(local_board, depth - 1, -beta, -alpha, root_depth);
                    rr.score = child_score;

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

        int best_score = -INF;
        std::optional<int> best_move = std::nullopt;

        for (auto &fut : futures) {
            try {
                RootEvalResult rr = fut.get();
                // callback for logging if requested
                try { if (on_move) on_move(rr); } catch(...) {}
                try { this->nodes += rr.nodes; } catch(...) {}
                if (rr.seldepth > this->max_seldepth) this->max_seldepth = rr.seldepth;
                if (rr.score > best_score) {
                    best_score = rr.score;
                    best_move = rr.move;
                    // store best child in plain TT
                    TTEntry e; e.key = k; e.depth = depth; e.score = best_score; e.flag = TTFlag::EXACT; e.best_move = best_move;
                    shared_tt_plain_store(k, e);
                }
            } catch(...) {
                // ignore individual failure
            }
            if (should_abort()) {
                if (abort_flag) abort_flag->store(true);
                break;
            }
        }

        if (!best_move.has_value()) best_score = evaluate_terminal(board);
        return best_score;
    }

    // ------------------ Seldepth Extensions ------------------
    inline int immediate_win(Board& board, int mv, const std::string& player) {
        try {
            board.make_move(mv);
            bool win = board.is_win(player);
            board.unmake_move(mv);
            return win ? 1 : 0;
        } catch(...) { return 0; }
    }

    inline int pv_move(const std::optional<int>& tt_move, int mv) {
        return (tt_move && *tt_move == mv) ? 1 : 0;
    }

    inline int near_endgame(Board& board) {
        int empty = board.legal_moves().size();
        return (empty <= 2) ? 2 : 0;
    }

    inline int quiescence(Board& board, const std::string& player) {
        // fully resolve "critical" positions in bigger boards
        try {
            auto moves = board.legal_moves();
            for (int mv : moves) {
                board.make_move(mv);
                if (board.is_win(player)) {
                    board.unmake_move(mv);
                    return 1;
                }
                board.unmake_move(mv);
            }
        } catch(...) {}
        return 0;
    }

    // ---------- negamax with PVS + LMR -------------------------------------
    int Searcher::negamax(Board& board, int depth, int alpha, int beta, int root_depth) {
        ++nodes;
        PlyGuard pg(this);

        if (should_abort()) return 0;

        uint64_t k = key(board);
        TTEntry entry;
        if (shared_tt_plain_get(k, entry) && entry.depth >= depth) {
            if (entry.flag == TTFlag::EXACT) return entry.score;
            if (entry.flag == TTFlag::LOWER && entry.score >= beta) return entry.score;
            if (entry.flag == TTFlag::UPPER && entry.score <= alpha) return entry.score;
        }

        // terminal / quiescence (board small so evaluate directly at depth 0)
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

        // order moves
        std::vector<int> ordered = order_moves_for_negamax(this, board, moves, k, depth);

        std::optional<int> tt_move_local = std::nullopt;
        if (shared_tt_plain_get(k, entry)) tt_move_local = entry.best_move;

        int best_score = -INF;
        std::optional<int> best_move = std::nullopt;
        int original_alpha = alpha;

        bool first = true;
        int move_index = 0;
        for (int mv : moves) {
            if (should_abort()) return 0;
            board.make_move(mv);
            int eval = -negamax(board, depth - 1, -beta, -alpha, root_depth);
            board.unmake_move(mv);
            int maxEval = std::numeric_limits<int>::min();
            if (eval > maxEval) maxEval = eval;
            if (eval > alpha) alpha = eval;
            if (alpha >= beta) break;
        }
        for (int mv : ordered) {
            if (should_abort()) {
                if (abort_flag) abort_flag->store(true);
                break;
            }
            ++move_index;
            try { board.make_move(mv); } catch(...) { continue; }

            int ext = 0;
            ext += immediate_win(board, mv, board.get_side_to_move());
            ext += pv_move(tt_move_local, mv);
            ext += near_endgame(board);
            ext += quiescence(board, board.get_side_to_move());

            int new_depth = depth - 1 + ext;

            int score;
            if (first) {
                // full window for first move
                score = -negamax(board, new_depth - 1, -beta, -alpha, root_depth);
            } else {
                // LMR + PVS: do a reduced/zero window search then re-search if it raises alpha
                int reduction = lmr_reduction(depth, move_index);
                int reduced_depth = std::max(0, new_depth - 1 - reduction);
                // try reduced (or null-window) search
                score = -negamax(board, reduced_depth, -alpha - 1, -alpha, root_depth);
                if (score > alpha && score < beta) {
                    // re-search full depth
                    score = -negamax(board, new_depth - 1, -beta, -alpha, root_depth);
                }
            }

            try { board.unmake_move(mv); } catch(...) {}

            first = false;

            if (score > best_score) { best_score = score; best_move = mv; }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                // fail-high -> update killer & history
                auto &kms = killer_moves[depth];
                if (std::find(kms.begin(), kms.end(), mv) == kms.end()) { kms.push_back(mv); if (kms.size() > 2) kms.erase(kms.begin()); }
                history[mv] += (1 << depth);

                TTEntry e; e.key = k; e.depth = depth; e.score = best_score; e.flag = TTFlag::LOWER; e.best_move = best_move;
                shared_tt_plain_store(k, e);
                return best_score;
            }
        }

        TTFlag final_flag;
        if (best_score <= original_alpha) final_flag = TTFlag::UPPER;
        else if (best_score >= beta) final_flag = TTFlag::LOWER;
        else final_flag = TTFlag::EXACT;

        TTEntry e; e.key = k; e.depth = depth; e.score = best_score; e.flag = final_flag; e.best_move = best_move;
        shared_tt_plain_store(k, e);
        return best_score;
    }

    // ---------- minimax root (parallelized per-root-move) --------------------
    // Note: added optional callback parameter `on_move` so callers (search()) can
    // receive per-root-move evaluation records for hybrid logging. Default null -> no callback.
    int Searcher::minimax_root(Board& board, int depth, int alpha, int beta, int root_depth, std::shared_ptr<ThreadPool> pool, std::function<void(const RootEvalResult&)> on_move) {
        ++nodes;
        PlyGuard pg(this);

        std::string root_player = board.get_side_to_move();
        uint64_t key_plain_local = key(board);
        uint64_t tk = make_root_key(key_plain_local, root_player.empty() ? 'X' : root_player[0]);

        auto moves = board.legal_moves();
        if (moves.empty()) return evaluate_for_root(board, root_player);

        std::optional<int> tt_move = std::nullopt;
        {
            TTEntry e;
            if (shared_tt_root_get(tk, e)) tt_move = e.best_move;
        }

        // Separate instant-winning moves first
        std::vector<int> winning, others;
        for (int mv : moves) {
            bool is_win = false;
            try {
                board.make_move(mv);
                std::string prev_player = board.get_side_to_move() == std::string(1, SYMBOL_X) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X);
                is_win = board.is_win(prev_player);
                board.unmake_move(mv);
            } catch(...) { /* treat as non-winning */ }
            if (is_win) winning.push_back(mv); else others.push_back(mv);
        }

        std::vector<int> combined;
        combined.insert(combined.end(), winning.begin(), winning.end());
        combined.insert(combined.end(), others.begin(), others.end());

        // order
        std::vector<int> ordered = order_moves_for_minimax(this, board, combined, tk, depth);

        std::vector<std::future<RootEvalResult>> futures;
        futures.reserve(ordered.size());

        for (int mv : ordered) {
            futures.emplace_back(pool->submit([this, board, mv, depth, alpha, beta, root_player, root_depth, tk]() -> RootEvalResult {
                RootEvalResult rr;
                rr.move = mv;
                rr.nodes = 0;
                rr.seldepth = 0;
                rr.score = -INF;
                try {
                    Board local_board = board; // copy
                    try { local_board.make_move(mv); } catch(...) { return rr; }

                    Searcher s_local;
                    s_local.abort_flag = this->abort_flag;
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

        int best_score = -INF;
        std::optional<int> best_move = std::nullopt;

        for (auto &fut : futures) {
            try {
                RootEvalResult rr = fut.get();
                // callback for logging if requested
                try { if (on_move) on_move(rr); } catch(...) {}
                try { this->nodes += rr.nodes; } catch(...) {}
                if (rr.seldepth > this->max_seldepth) this->max_seldepth = rr.seldepth;
                if (rr.score > best_score) {
                    best_score = rr.score;
                    best_move = rr.move;
                    TTEntry e; e.key = tk; e.depth = depth; e.score = best_score; e.flag = TTFlag::EXACT; e.best_move = best_move;
                    shared_tt_root_store(tk, e);
                }
            } catch(...) {}
            if (should_abort()) {
                if (abort_flag) abort_flag->store(true);
                break;
            }
        }

        if (!best_move.has_value()) best_score = evaluate_for_root(board, root_player);
        return best_score;
    }

    // ---------- minimax with ordering + LMR-like early reductions -----------
    int Searcher::minimax(Board& board, int depth, int alpha, int beta, const std::string& root_player, int root_depth) {
        ++nodes;
        PlyGuard pg(this);

        if (should_abort()) {
            if (abort_flag) abort_flag->store(true);
            return 0;
        }

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

        // separate winning moves quickly to prioritize them
        std::vector<int> winning, others;
        for (int mv : moves) {
            bool is_win = false;
            try {
                board.make_move(mv);
                std::string prev_player = board.get_side_to_move() == std::string(1, SYMBOL_X) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X);
                is_win = board.is_win(prev_player);
                board.unmake_move(mv);
            } catch(...) {}
            if (is_win) winning.push_back(mv); else others.push_back(mv);
        }
        std::vector<int> combined;
        combined.insert(combined.end(), winning.begin(), winning.end());
        combined.insert(combined.end(), others.begin(), others.end());

        std::vector<int> ordered = order_moves_for_minimax(this, board, combined, tk, depth);

        bool maximizing = (board.get_side_to_move() == root_player);
        int best_score = maximizing ? -INF : INF;
        std::optional<int> best_move = std::nullopt;
        int original_alpha = alpha, original_beta = beta;

        int move_idx = 0;
        for (int mv : ordered) {
            if (should_abort()) {
                if (abort_flag) abort_flag->store(true);
                break;
            }
            ++move_idx;
            try { board.make_move(mv); } catch(...) { continue; }

            std::optional<int> tt_move_local = std::nullopt;
            int ext = 0;
            ext += immediate_win(board, mv, board.get_side_to_move());
            ext += pv_move(tt_move_local, mv);
            ext += near_endgame(board);
            ext += quiescence(board, board.get_side_to_move());

            int new_depth = depth - 1 + ext;

            int score;
            // Use LMR-like reduction (only on the side that is not immediate-winning)
            int reduction = lmr_reduction(depth, move_idx);
            int search_depth = std::max(0, depth - 1 - reduction);

            // null-window for non-first moves and non-winning moves, then re-search if needed
            if (move_idx == 1) {
                score = minimax(board, new_depth - 1, alpha, beta, root_player, root_depth);
            } else {
                if (maximizing) {
                    score = minimax(board, search_depth, alpha, alpha + 1, root_player, root_depth);
                    if (score > alpha && score < beta) {
                        score = minimax(board, new_depth - 1, alpha, beta, root_player, root_depth);
                    }
                } else {
                    score = minimax(board, search_depth, beta - 1, beta, root_player, root_depth);
                    if (score < beta && score > alpha) {
                        score = minimax(board, new_depth - 1, alpha, beta, root_player, root_depth);
                    }
                }
            }

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
                return best_score;
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

        // Best-effort mirror store to plain TT
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
                try { board.make_move(mv); } catch(...) { break; }
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
                try { cur.make_move(mv); } catch(...) { break; }
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
