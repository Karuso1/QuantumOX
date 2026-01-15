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
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric> // for accumulate
#include <optional>
#include <queue>
#include <shared_mutex> // for shared mutex
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
            val = board.evaluate(for_player[0]);
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

    // small helper for order_moves_for_negamax
    static inline double safe_log(double x) {
        return x <= 0.0 ? 0.0 : std::log1p(x);
    }

    // ---------- Utility: order moves with advanced heuristics ---------------
    // returns vector of moves sorted (best first) but DOES NOT modify board.
    std::vector<int> order_moves_for_negamax(Searcher* s, Board& board, std::vector<int> moves, uint64_t k, int depth) {
        struct MoveKey {
            int mv;
            double key;
            size_t orig;
            int reasons[10];
        };

        // Trivial cases
        if (moves.size() <= 1) return moves;

        std::vector<MoveKey> mk; mk.reserve(moves.size());

        // ------------------ Configurable thresholds & weights ------------------
        const size_t BF = moves.size();
        const bool very_wide = (BF > 120);    // skip almost all probes
        const bool shallow = (depth <= 3);    // allow more tactical checks in shallow nodes
        const size_t probe_limit = 64;        // only probe when BF <= this
        const size_t mobility_limit = 128;

        // Weight buckets (larger is more important)
        const double W_TT = 1e8;           // transposition-table exact move
        const double W_PV = 6e7;           // principal-variation / searcher-selected PV
        const double W_IMMEDIATE = 5e7;    // immediate win
        const double W_THREAT = 2e5;       // threat creation (fork / double-threat)
        const double W_BLOCK = 9e4;        // move blocks opponent immediate threat
        const double W_KILLER = 5e4;       // killer move boost
        const double W_HISTORY = 8.0;      // history heuristic scaled
        const double W_MOBILITY = 1.5;     // mobility (availability of responses)
        const double W_STABILITY = 3e4;    // stable moves (don't create opponent fork)
        const double W_SYMMETRY = 120.0;   // prefer symmetric/paired moves when relevant
        const double W_CENTER = 160.0;     // square center bias
        const double W_CORNER = 90.0;      // corner bias
        const double W_EDGE = 18.0;        // edge bias
        const double tiny_tiebreak = 1.0 / 1e7;

        // ------------------ read TT + PV-ish information -----------------------
        // Try plain TT keyed by k (this function expects plain-keyed lookup)
        std::optional<int> tt_move = std::nullopt;
        auto ittt = s->tt_plain.find(k);
        if (ittt != s->tt_plain.end()) {
            if (ittt->second.best_move) tt_move = ittt->second.best_move;
        }

        // Try to get a candidate PV from Searcher::build_pv (cheap-ish, but only use if short)
        std::vector<int> engine_pv;
        try {
            // build_pv is public; it's cheap relative to expensive probes
            engine_pv = s->build_pv_for_root(board, board.get_side_to_move());
        } catch(...) {
            engine_pv.clear();
        }
        std::optional<int> pv_first = std::nullopt;
        if (!engine_pv.empty()) pv_first = engine_pv.front();

        // -------------- per-call cache to avoid redoing probes -----------------
        std::unordered_map<uint64_t, double> probe_cache; probe_cache.reserve(std::min<size_t>(BF, 128));

        // Helper: positional bias (square grids)
        auto pos_bias = [&](const Board& b, int mv) -> double {
            auto dims = b.get_dims();
            if (dims.size() != 2) return 0.0;
            int N = dims[0];
            if (N <= 0 || dims[1] != N) return 0.0;
            int pos = mv - 1;
            int r = pos / N, c = pos % N;
            if (N % 2 == 1 && r == N/2 && c == N/2) return W_CENTER;
            if ((r==0 || r==N-1) && (c==0 || c==N-1)) return W_CORNER;
            if (r==0 || r==N-1 || c==0 || c==N-1) return W_EDGE;
            return 0.0;
        };

        // Helper: small mobility metric (how many replies available after move)
        auto mobility_score = [&](Board& b, int mv)->double {
            if (BF > mobility_limit) return 0.0; // skip in extreme wide nodes
            try {
                std::string before = b.get_side_to_move();
                b.make_move(mv);
                size_t replies = b.legal_moves().size();
                b.unmake_move(mv);
                // more replies for opponent -> dangerous (we penalize), fewer replies -> good (we reward)
                // invert replies with smooth transform
                double inv = 1.0 / (1.0 + static_cast<double>(replies));
                return W_MOBILITY * inv * 100.0;
            } catch(...) {
                return 0.0;
            }
        };

        // Helper: cheap fork/threat detector using one-move lookahead (only when not wide)
        // returns (immediate_win, creates_threat_count, blocks_opponent_win)
        auto cheap_tactical = [&](Board& b, int mv)->std::tuple<bool,int,bool> {
            uint64_t ck = (static_cast<uint64_t>(k) << 32) ^ static_cast<uint64_t>(mv & 0xffffffff);
            auto itc = probe_cache.find(ck);
            if (itc != probe_cache.end()) {
                double v = itc->second;
                bool win = (std::floor(v / 1e9) > 0.0);
                int threats = static_cast<int>(std::fmod(std::floor(v / 1e5), 1e4));
                bool block = (std::fmod(v, 1e5) >= 1.0);
                return {win, threats, block};
            }
        
            bool immediate = false;
            int threats = 0;
            bool blocks = false;
            try {
                std::string before = b.get_side_to_move();
                b.make_move(mv);
            
                // immediate win?
                if (b.is_win(before)) {
                    immediate = true;
                    // encode and cache
                    probe_cache[ck] = 1e9 + 0.0;
                    b.unmake_move(mv);
                    return {true,0,false};
                }
            
                // count moves that would be immediate wins for the mover on the following ply (fork-like)
                auto next = b.legal_moves();
                size_t cap = (next.size() > 64) ? 64 : next.size();
                for (size_t i = 0; i < cap; ++i) {
                    int nm = next[i];
                    try {
                        b.make_move(nm);
                        if (b.is_win(before)) threats++;
                        b.unmake_move(nm);
                        if (threats >= 2) break; // fork detection
                    } catch(...) { }
                }
            
                // block: check if opponent had an immediate winning reply in the root position that is now removed
                bool opponent_had_immediate = false;
                // cheap scan on root position (only if small)
                if (BF <= probe_limit) {
                    auto root_moves = board.legal_moves(); // original root board
                    for (int rm : root_moves) {
                        try {
                            board.make_move(rm);
                            std::string opp = board.get_side_to_move();
                            if (board.is_win(opp == std::string(1, 'X') ? std::string(1, 'O') : std::string(1, 'X'))) {
                                opponent_had_immediate = true;
                                board.unmake_move(rm);
                                break;
                            }
                            board.unmake_move(rm);
                        } catch(...) {}
                    }
                }
                // Now see if the new board removed that immediate (cheap approx)
                if (opponent_had_immediate) {
                    // check if any opponent wins exist now
                    bool opponent_win_now = false;
                    for (int nm : b.legal_moves()) {
                        try {
                            b.make_move(nm);
                            std::string opp = b.get_side_to_move();
                            if (b.is_win(opp == std::string(1, 'X') ? std::string(1, 'O') : std::string(1, 'X'))) {
                                opponent_win_now = true;
                                b.unmake_move(nm);
                                break;
                            }
                            b.unmake_move(nm);
                        } catch(...) {}
                    }
                    if (!opponent_win_now) blocks = true;
                }
            
                // encode: 1e9 block for immediate, add threats*1e5, set last digits as block flag
                double enc = 0.0;
                if (immediate) enc += 1e9;
                enc += static_cast<double>(threats) * 1e5;
                if (blocks) enc += 1.0;
                probe_cache[ck] = enc;
                b.unmake_move(mv);
            } catch(...) {
                // best-effort, keep defaults
            }
            return {immediate, threats, blocks};
        };

        // ------------------ main evaluation loop --------------------
        for (size_t i = 0; i < moves.size(); ++i) {
            int mv = moves[i];
            MoveKey M; M.mv = mv; M.orig = i; M.key = 0.0;
            for (int r=0;r<10;++r) M.reasons[r] = 0;
        
            // 1) TT boost (plain)
            if (tt_move && *tt_move == mv) {
                M.key += W_TT;
                M.reasons[0] = 90000000;
            }
        
            // 2) PV / engine-suggested first move (highly prioritized, different weight from TT)
            if (pv_first && *pv_first == mv) {
                M.key += W_PV;
                M.reasons[1] = 60000000;
            }
        
            // 3) killer moves
            auto itkm = s->killer_moves.find(depth);
            if (itkm != s->killer_moves.end()) {
                auto &klist = itkm->second;
                if (!klist.empty() && klist[0] == mv) { M.key += W_KILLER; M.reasons[2] = 50000; }
                else if (klist.size() > 1 && klist[1] == mv) { M.key += (W_KILLER * 0.6); M.reasons[2] = 30000; }
            }
        
            // 4) history heuristic (logarithmic scaling)
            auto ith = s->history.find(mv);
            if (ith != s->history.end()) {
                double h = static_cast<double>(ith->second);
                double hist_score = W_HISTORY * safe_log(std::abs(h) + 1.0);
                if (h < 0) hist_score = -hist_score;
                M.key += hist_score;
                M.reasons[3] = static_cast<int>(hist_score);
            }
        
            // 5) positional bias (center/corner/edge)
            double pb = pos_bias(board, mv);
            M.key += pb;
            M.reasons[4] = static_cast<int>(pb);
        
            // 6) mobility / replies scoring (prefer moves that reduce opponent mobility)
            double mob = mobility_score(board, mv);
            M.key += mob;
            M.reasons[5] = static_cast<int>(mob);
        
            // 7) symmetry/pairing bias: if grid even and move has a symmetric counterpart not played, favor it
            // (helps keep balanced play and reduces branching near symmetric positions)
            auto dims = board.get_dims();
            if (dims.size() == 2 && dims[0] == dims[1] && dims[0] >= 2) {
                int N = dims[0];
                int pos = mv - 1, r = pos / N, c = pos % N;
                int sr = N - 1 - r, sc = N - 1 - c;
                int sym_pos = sr * N + sc + 1; // 1-based
                if (sym_pos != mv) {
                    // if symmetric square is still legal, give small boost
                    auto legal = board.legal_moves();
                    if (std::find(legal.begin(), legal.end(), sym_pos) != legal.end()) {
                        M.key += W_SYMMETRY;
                        M.reasons[6] = 1;
                    }
                }
            }
        
            // 8) cheap tactical probe (immediate win, threats, block) - gated by width/depth
            if (!very_wide && (shallow || BF <= probe_limit)) {
                bool immediate=false; int threats=0; bool blocks=false;
                std::tie(immediate, threats, blocks) = cheap_tactical(board, mv);
                if (immediate) {
                    M.key += W_IMMEDIATE;
                    M.reasons[7] = 10000000;
                }
                if (threats >= 1) {
                    M.key += static_cast<double>(threats) * W_THREAT;
                    M.reasons[8] = threats;
                }
                if (blocks) {
                    M.key += W_BLOCK;
                    M.reasons[9] = 1;
                }
            } else {
                // in very wide nodes, do a super cheap replacement: check immediate win only (safe)
                try {
                    std::string before = board.get_side_to_move();
                    board.make_move(mv);
                    if (board.is_win(before)) {
                        M.key += (W_IMMEDIATE * 0.8);
                        M.reasons[7] = 8000000;
                    }
                    board.unmake_move(mv);
                } catch(...) {}
            }
        
            // 9) stability: penalize moves that create many opponent immediate threats
            if (!very_wide && BF <= probe_limit) {
                try {
                    std::string before = board.get_side_to_move();
                    board.make_move(mv);
                    int opp_threats = 0;
                    auto next = board.legal_moves();
                    size_t cap = (next.size() > 64) ? 64 : next.size();
                    for (size_t j=0;j<cap;++j) {
                        int nm = next[j];
                        try {
                            board.make_move(nm);
                            if (board.is_win(before == std::string(1, 'X') ? std::string(1, 'O') : std::string(1, 'X'))) {
                                opp_threats++;
                            }
                            board.unmake_move(nm);
                            if (opp_threats >= 2) break;
                        } catch(...) {}
                    }
                    board.unmake_move(mv);
                    if (opp_threats == 0) {
                        M.key += W_STABILITY;
                    } else {
                        M.key -= static_cast<double>(opp_threats) * (W_STABILITY * 0.6);
                    }
                } catch(...) {}
            }
        
            // 10) tiny deterministic bias for stability of sort ordering
            M.key += -static_cast<double>(mv) * tiny_tiebreak;
        
            mk.push_back(M);
        }

        // ------------------ Sort: highest key first, stable by original index -------------
        std::sort(mk.begin(), mk.end(), [](const MoveKey& a, const MoveKey& b){
            if (a.key == b.key) return a.orig < b.orig;
            return a.key > b.key;
        });

        std::vector<int> out; out.reserve(mk.size());
        for (const auto &m : mk) out.push_back(m.mv);
        return out;
    }

    // small helpers for order_moves_for_minimax
    static inline double clamp_double(double x, double lo, double hi) {
        if (x < lo) return lo;
        if (x > hi) return hi;
        return x;
    }

    // scale an int to double
    static inline double to_d(int x) { return static_cast<double>(x); }

    std::vector<int> order_moves_for_minimax(Searcher* s, Board& board, std::vector<int> moves, uint64_t tk, int depth) {
        struct MoveKey { int mv; double key; size_t orig; int components[8]; };

        // Early out: trivial ordering when there is 0 or 1 move
        if (moves.size() <= 1) return moves;

        std::vector<MoveKey> mk; mk.reserve(moves.size());

        // --------------- retrieve TT best-move (root keyed) if present --------------
        std::optional<int> tt_move = std::nullopt;
        // prefer root-keyed table (more accurate for ordering at root)
        auto itroot = s->tt_root.find(tk);
        if (itroot != s->tt_root.end()) {
            if (itroot->second.best_move) tt_move = itroot->second.best_move;
        } else {
            // fallback: plain-keyed TT lookup using board key (mix with tk)
            uint64_t bkey = s->key(board);
            auto itplain = s->tt_plain.find(bkey);
            if (itplain != s->tt_plain.end()) {
                if (itplain->second.best_move) tt_move = itplain->second.best_move;
            }
        }

        // -------------- small bookkeeping and thresholds ----------------
        const size_t BF = moves.size();
        // If branching is large, skip expensive per-move probes
        const bool wide_node = (BF > 60); // tunable threshold
        // If depth is shallow, prioritize tactical checks (mates/forks)
        const bool shallow_node = (depth <= 3);
        // cap for performing tactical probes to avoid blowups
        const size_t tactical_probe_limit = 48; // only probe when BF <= this

        // per-call cache to avoid repeating Searcher evaluations
        // key: (tk << 32) ^ mv  => careful mixing where shifts safe
        std::unordered_map<uint64_t, double> eval_cache; eval_cache.reserve(std::min<size_t>(BF, 128));

        // string representing side to move before any make_move
        std::string root_player = board.get_side_to_move();

        // small penalty for larger move numbers to keep deterministic tiebreak
        const double base_tiebreak = 1.0 / 1e6;

        // access killer moves and history references (cheap)
        auto it_killer = s->killer_moves.find(depth);
        const std::vector<int> empty_killers;
        const std::vector<int>& killers = (it_killer != s->killer_moves.end()) ? it_killer->second : empty_killers;

        // history score: move -> int
        const auto& history = s->history;

        // --------------- weight constants (tunable) ------------------
        // Use large differences so categories don't bleed
        const double W_TT = 1e9;
        const double W_IMMEDIATE_WIN = 5e8;
        const double W_KILLER_FIRST = 6e4;
        const double W_KILLER_SECOND = 2e4;
        const double W_HISTORY = 10.0;     // scaled by history value
        const double W_EVAL = 2e3;         // scaled by evaluation magnitude
        const double W_CENTER = 150.0;
        const double W_CORNER = 75.0;
        const double W_EDGE = 12.0;
        const double W_FORK = 2e5;
        const double W_BLOCK = 5e4;

        // helper to detect center/corner/edge for square grids
        auto positional_bias = [&](const Board& b, int mv)->double {
            const auto dims = b.get_dims();
            if (dims.size() != 2) return 0.0;
            int N = dims[0];
            if (N <= 0 || dims[1] != N) return 0.0; // only square
            int pos = mv - 1; // moves are 1-based in the engine
            int r = pos / N;
            int c = pos % N;
            // center
            if (N % 2 == 1 && r == N/2 && c == N/2) return W_CENTER;
            // corner
            if ((r==0 || r==N-1) && (c==0 || c==N-1)) return W_CORNER;
            // edge
            if (r==0 || r==N-1 || c==0 || c==N-1) return W_EDGE;
            return 0.0;
        };

        // helper: fast probe performing a small tactical check after make_move
        // returns tuple: <is_immediate_win, is_fork_like, eval_score>
        auto tactical_probe = [&](Board& b, int mv)->std::tuple<bool,bool,double> {
            // key for cache
            uint64_t ck = ((uint64_t)tk << 32) ^ static_cast<uint64_t>(mv & 0xffffffff);
            auto itc = eval_cache.find(ck);
            if (itc != eval_cache.end()) {
                double v = itc->second;
                bool win = (v >= 1e8);
                bool fork = (v >= 1e5 && v < 1e8);
                double eval = (v >= 1e5) ? (v - (int)1e5) : v; // encoding scheme if used
                return {win, fork, eval};
            }

            bool immediate_win = false;
            bool fork_like = false;
            double eval_score = 0.0;

            try {
                // Save minimal info: root player string before move
                std::string before = b.get_side_to_move();
                b.make_move(mv);

                // immediate win for the player who moved?
                if (b.is_win(before)) {
                    immediate_win = true;
                    // cache with a large sentinel
                    eval_cache[ck] = W_IMMEDIATE_WIN + 1.0; // encode as > W_IMMEDIATE_WIN
                    b.unmake_move(mv);
                    return {true,false,0.0};
                }

                // lightweight evaluation: call Searcher::evaluate_for_root if available
                // It's an engine-provided heuristic evaluator: use it sparingly.
                try {
                    eval_score = static_cast<double>(s->evaluate_for_root(b, before));
                } catch(...) {
                    eval_score = 0.0;
                }

                // fork detection: count number of immediate winning moves available next turn for 'before'
                size_t win_moves = 0;
                if (b.legal_moves().size() <= 64) { // limit work
                    auto next_moves = b.legal_moves();
                    for (int nm : next_moves) {
                        try {
                            b.make_move(nm);
                            if (b.is_win(before)) {
                                ++win_moves;
                            }
                            b.unmake_move(nm);
                            if (win_moves >= 2) break; // fork found
                        } catch(...) {
                            // ignore bad/unexpected
                            if (win_moves >= 2) break;
                        }
                    }
                }
                if (win_moves >= 2) fork_like = true;

                // encode results in cache: we compress into a double value for simplicity
                double enc = eval_score;
                if (immediate_win) enc += 1e8; // sentinel
                else if (fork_like) enc += 1e5; // sentinel
                eval_cache[ck] = enc;

                b.unmake_move(mv);
            } catch(...) {
                // if any make/unmake failed, be conservative and set neutral values
                immediate_win = false;
                fork_like = false;
                eval_score = 0.0;
                // attempt to restore board if possible is omitted here
            }

            return {immediate_win, fork_like, eval_score};
        };


        // ------------------- main scoring loop ---------------------
        for (size_t i = 0; i < moves.size(); ++i) {
            int mv = moves[i];
            MoveKey m{}; m.mv = mv; m.orig = i;
            m.key = 0.0;
            for (int k=0;k<8;++k) m.components[k]=0;

            // 1) TT move gets top score
            if (tt_move && *tt_move == mv) {
                m.key += W_TT;
                m.components[0] = 1000000000;
            }

            // 2) killer moves: 1st killer > 2nd killer > others
            if (!killers.empty()) {
                if (killers.size() > 0 && killers[0] == mv) { m.key += W_KILLER_FIRST; m.components[1] = 60000; }
                else if (killers.size() > 1 && killers[1] == mv) { m.key += W_KILLER_SECOND; m.components[1] = 20000; }
            }

            // 3) history heuristic
            auto ith = history.find(mv);
            if (ith != history.end()) {
                // scale history logarithmically to avoid domination
                double h = static_cast<double>(ith->second);
                double hist_score = W_HISTORY * std::log1p(1.0 + std::abs(h));
                if (h < 0) hist_score = -hist_score; // preserve sign
                m.key += hist_score;
                m.components[2] = static_cast<int>(hist_score);
            }

            // 4) positional cheap bias (center/corner/edge)
            m.key += positional_bias(board, mv);
            m.components[3] = static_cast<int>(positional_bias(board, mv));

            // 5) tiny deterministic tiebreak to ensure consistent ordering
            m.key += -static_cast<double>(mv) * base_tiebreak;

            // 6) lightweight tactical probes are only run when the node is not very wide
            if (!wide_node && (shallow_node || BF <= tactical_probe_limit)) {
                bool immediate_win = false;
                bool fork_like = false;
                double eval_score = 0.0;
                std::tie(immediate_win, fork_like, eval_score) = tactical_probe(board, mv);

                if (immediate_win) {
                    m.key += W_IMMEDIATE_WIN;
                    m.components[4] = 100000000;
                } else {
                    if (fork_like) {
                        m.key += W_FORK;
                        m.components[5] = 200000;
                    }
                    // evaluation: positive means better for root_player
                    // we add scaled eval; clamp to reduce dominance
                    double scaled_eval = W_EVAL * clamp_double(eval_score / 1000.0, -50.0, 50.0);
                    m.key += scaled_eval;
                    m.components[6] = static_cast<int>(scaled_eval);
                }
            }

            // 7) block detection: does this move prevent an opponent immediate win?
            // We use a cheap probe: simulate opponent's best reply to see if there's an immediate win blocked
            if (!wide_node && BF <= tactical_probe_limit) {
                try {
                    std::string before = board.get_side_to_move();
                    bmake:
                    board.make_move(mv);
                    // Now check opponent:
                    auto next = board.legal_moves();
                    bool opponent_has_win = false;
                    for (int nm : next) {
                        board.make_move(nm);
                        if (board.is_win(before == std::string(1, SYMBOL_X) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X))) {
                            opponent_has_win = true;
                            board.unmake_move(nm);
                            break;
                        }
                        board.unmake_move(nm);
                    }
                    board.unmake_move(mv);
                    if (!opponent_has_win) {
                        // if opponent previously had a direct win and this move removes it, award block
                        // This is a heuristic and requires more engine knowledge to be exact; we approximate
                        // by re-checking the root position: did opponent have an immediate win before mv?
                        bool opponent_had_win_before = false;
                        // Quick check: look at root legal moves and see if any gave opponent a win
                        // (only do this if BF is not huge)
                        if (BF <= tactical_probe_limit) {
                            auto root_moves = board.legal_moves();
                            for (int rm : root_moves) {
                                board.make_move(rm);
                                if (board.is_win(board.get_side_to_move() == std::string(1, SYMBOL_X) ? std::string(1, SYMBOL_O) : std::string(1, SYMBOL_X))) {
                                    opponent_had_win_before = true;
                                    board.unmake_move(rm);
                                    break;
                                }
                                board.unmake_move(rm);
                            }
                        }
                        if (opponent_had_win_before) {
                            m.key += W_BLOCK;
                            m.components[7] = 50000;
                        }
                    }
                } catch(...) {
                    // If anything goes wrong just skip block detection
                }
            }

            mk.push_back(m);
        }

        // ---------------- final sort: by key desc, stable with orig index as tiebreaker ----------------
        std::sort(mk.begin(), mk.end(), [](const MoveKey& a, const MoveKey& b){
            if (a.key == b.key) return a.orig < b.orig; // stable
            return a.key > b.key;
        });

        // build output vector of moves
        std::vector<int> out; out.reserve(mk.size());
        for (const auto &m : mk) out.push_back(m.mv);

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
                info_token("minimaxpv");
                for (int mv : min_pv) info_token(std::to_string(mv));
                info_token("negamaxpv");
                for (int mv : neg_pv) info_token(std::to_string(mv));
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
                            
                size_t max_moves = std::max(neg_root_results.size(), min_root_results.size());
                if (max_moves == 0) max_moves = 0; // safe guard
                            
                std::vector<double> selected_scores_minimax;
                std::vector<double> selected_scores_negamax;
                            
                std::unordered_set<int> printed_moves; // track already printed moves
                size_t currmovenumber = 1;             // sequential numbering for unique moves
                            
                for (size_t idx = 0; idx < max_moves; ++idx) {
                
                    // Decide which algorithm to use for this currmovenumber
                    int mm_score = (idx < min_root_results.size()) ? min_root_results[idx].score : -INF;
                    int nn_score = (idx < neg_root_results.size()) ? neg_root_results[idx].score : -INF;
                
                    std::string chosen_alg_for_index = "minimax";
                    if (nn_score > mm_score) chosen_alg_for_index = "negamax";
                    else if (mm_score > nn_score) chosen_alg_for_index = "minimax";
                    else chosen_alg_for_index = "minimax"; // tie -> prefer minimax
                
                    // select move to print
                    int currmove = 0;
                    if (chosen_alg_for_index == "minimax") {
                        if (idx < min_root_results.size()) {
                            currmove = min_root_results[idx].move;
                            selected_scores_minimax.push_back(static_cast<double>(min_root_results[idx].score));
                        }
                    } else {
                        if (idx < neg_root_results.size()) {
                            currmove = neg_root_results[idx].move;
                            selected_scores_negamax.push_back(static_cast<double>(neg_root_results[idx].score));
                        }
                    }
                
                    // only print if not already printed
                    if (printed_moves.find(currmove) == printed_moves.end()) {
                        printed_moves.insert(currmove);
                        std::cout << "info currmove " << currmove << " currmovenumber " << currmovenumber << std::endl;
                        currmovenumber++; // increment only for unique moves
                    }
                
                    if (should_abort()) break;
                
                    // compute overall for chosen algorithm (kept for logic)
                    double overall = 0.0;
                    if (chosen_alg_for_index == "minimax") {
                        if (!selected_scores_minimax.empty()) {
                            overall = std::accumulate(selected_scores_minimax.begin(), selected_scores_minimax.end(), 0.0) / selected_scores_minimax.size();
                        }
                    } else {
                        if (!selected_scores_negamax.empty()) {
                            overall = std::accumulate(selected_scores_negamax.begin(), selected_scores_negamax.end(), 0.0) / selected_scores_negamax.size();
                        }
                    }
                    [[maybe_unused]] long overall_rounded = static_cast<long>(std::floor(overall + 0.5));
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
                info_token("minimaxpv");
                for (int mv : min_pv) info_token(std::to_string(mv));
                info_token("negamaxpv");
                for (int mv : neg_pv) info_token(std::to_string(mv));
                info_token("pv");
                for (int mv : best_pv) info_token(std::to_string(mv));
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
        try { return board.evaluate(stm[0]); } catch(...) { return evaluate_terminal(board); }
    }

    int Searcher::evaluate_for_root(Board& board, const std::string& root_player) {
        try { return board.evaluate(root_player[0]); } catch(...) {
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
