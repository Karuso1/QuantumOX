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

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <utility>
#include <map>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <limits>
#include <queue>

#include "constants.h"
#include "utils.h"
#include "options.h"
#include "engine.h"

using namespace QuantumOX;

std::mutex input_mtx;
std::queue<std::string> input_queue;
std::condition_variable input_cv;
std::atomic<bool> running{true};
std::atomic<bool> search_running{false};
std::thread search_thread;

static const char* HELP_TEXT = R"(QuantumOX UTTTI commands and usage:

UTTTI Overview:
  QuantumOX uses the Universal Tic Tac Toe Interface (UTTTI), a protocol for communicating
  between the engine and a user interface or scripts. Commands are sent as text lines, and
  the engine responds with status, evaluation, or move information. UTTTI supports boards of
  different sizes and dimensions (e.g., 3x3, 4x4, 5x5, or 3D boards like 3x3x3).

Engine Options:
  - The engine supports configurable options that can change behavior, board size, player
    order, search threads, and more.
  - Options are set with 'setoption name <name> value <value>'.
  - Certain options, such as the board size (Grid), affect move numbering:
      - Moves always start at 1.
      - The maximum move number depends on the current grid size (e.g., 1-9 for 3x3, 1-16
        for 4x4, 1-27 for 3x3x3, etc.).

Commands:

  uttti
    - Performs a handshake with the engine to confirm UTTTI support.

  setoption name <name> value <value>
    - Sets engine options. Use exact option names recognized by the engine.
    - Options may affect engine behavior, board size, or search parameters.

  isready
    - Checks if the engine has finished initialization and is ready to receive commands.
      The engine responds with "readyok" when ready.

  utttinewgame
    - Resets the board state and internal search history silently. Use at the start of a new game.

  grid emptygrid [fill <moves> ...]
    - Resets the board to an empty grid.
    - Fill the board with a move sequence; the first move depends on the FirstPlayer option:
        grid emptygrid fill <first player's move> <second player's move> ...
      - Moves **start at 1** and increase left-to-right, top-to-bottom, 
        up to the total number of cells in the current grid.
      - The sequence alternates between the first player and the second player moves until the game ends.
      - Invalid moves below 1 will break UTTTI; moves above the current grid's maximum
        are valid only if the grid size has been increased using an option like Grid.

  go [depth <D>, movetime <M>, nodes <N>]
    - Starts a search using the specified limit (depth, time in ms, or nodes). Outputs
      principal variation, evaluation score, and nodes searched.

  stop
    - Immediately stops the current search and returns the best move found so far.

  quit / exit
    - Terminates the engine cleanly.

  help
    - Prints this help text.

Tips:
  - Commands are case-insensitive but follow the proper spacing format.
  - Use 'setoption' to configure engine behavior before starting a search.
  - 'grid emptygrid fill ...' is the preferred way to set up a full board state.
  - Moves always start at 1; the maximum depends on the current grid size.
  - UTTTI responses are designed for easy parsing by GUIs or scripts.

For advanced usage, refer to the QuantumOX documentation or UTTTI protocol specification.)";

static const std::map<std::string, std::string> COMMAND_HELPS = {
    {"grid", R"(Usage:
  grid emptygrid [fill <first player's move> <second player's move> ...]
Examples:
  grid emptygrid
  grid emptygrid fill 1 2 3)"},
    {"go", R"(Usage:
  go [depth <N>] [movetime <ms>|movetime_ms <ms>] [nodes <N>]
Examples:
  go depth 4
  go movetime 2000
  go nodes 10000)"},
    {"setoption", R"(Usage:
  setoption name <name> value <value>
Example:
  setoption name Hash value 32)"},
    {"uttti", "Performs handshake and lists engine options."},
    {"utttinewgame", "Indicates the engine that a new game has started."},
    {"help", "Usage: help [command]\nShow help for commands or a specific command."},
    {"stop", R"(Usage:
  stop
Description:
  Immediately stops the current search and returns the best move found so far.)"},
    {"quit", R"(Usage:
  quit / exit
Description:
  Terminates the engine cleanly and exits.)"},
    {"exit", "Alias for quit."},
    {"isready", R"(Usage:
  isready
Description:
  Checks if the engine has finished initialization and is ready to receive commands.
  The engine responds with 'readyok' when ready.)"}
};

// Valid flags/subcommands per command (lowercase)
static const std::map<std::string, std::vector<std::string>> VALID_TOKENS = {
    // grid: first token after "grid" is a subcommand; for emptygrid we accept the flag "fill"
    {"grid", {"emptygrid", "fill", "tttn"}},
    // go: recognized flags
    {"go", {"depth", "movetime", "movetime_ms", "nodes"}},
    // setoption expects tokens "name" and "value"
    {"setoption", {"name", "value"}},
    // uttti/help don't need flags but keep empty vector for completeness
    {"uttti", {}},
    {"utttinewgame", {}},
    {"help", {}}
};

// -- helper utilities --------------------------------------------------------

static std::string to_lower_copy(const std::string &s) {
    std::string t = s;
    for (auto &c : t) c = static_cast<char>(std::tolower(c));
    return t;
}

// Classic Levenshtein distance
static int levenshtein_distance(const std::string &a, const std::string &b) {
    size_t n = a.size();
    size_t m = b.size();
    if (n == 0) return static_cast<int>(m);
    if (m == 0) return static_cast<int>(n);
    std::vector<int> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = static_cast<int>(j);
    for (size_t i = 1; i <= n; ++i) {
        cur[0] = static_cast<int>(i);
        for (size_t j = 1; j <= m; ++j) {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            cur[j] = std::min({ prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost });
        }
        prev.swap(cur);
    }
    return prev[m];
}

// Suggest the closest valid token from a list, or empty if none is good
static std::string suggest_closest(const std::string &token, const std::vector<std::string> &candidates) {
    if (candidates.empty()) return "";
    std::string lower_token = to_lower_copy(token);
    int bestDist = std::numeric_limits<int>::max();
    std::string bestMatch;
    for (const auto &cand : candidates) {
        std::string lower_c = to_lower_copy(cand);
        int d = levenshtein_distance(lower_token, lower_c);
        if (d < bestDist) {
            bestDist = d;
            bestMatch = cand;
        }
    }
    // only suggest if reasonably close (heuristic)
    // allow suggestion if distance <= max(1, token.length()/3)
    int threshold = std::max(1, static_cast<int>(token.length()/3));
    if (bestDist <= threshold) return bestMatch;
    return "";
}

// check if token is in candidates (case-insensitive)
static bool token_in(const std::string &token, const std::vector<std::string> &candidates) {
    std::string t = to_lower_copy(token);
    for (const auto &c : candidates) if (to_lower_copy(c) == t) return true;
    return false;
}

// -- end helper utilities ----------------------------------------------------

void handle_uttti(QuantumOXEngine& engine) {
    (void)engine; // suppress unused parameter warning

    std::cout << "id name " << ENGINE_NAME << "\n";
    std::cout << "id author " << ENGINE_AUTHOR << "\n\n";

    auto opts = list_options();
    for (const auto& kv : opts) {
        const std::string& name = kv.first;
        const auto& meta = kv.second;

        std::ostringstream line;
        line << "option name " << name
             << " type " << meta.at("type");

        // if option has default, print it (for any type)
        if (meta.find("default") != meta.end())
            line << " default " << meta.at("default");

        // print other common params depending on type
        if (meta.at("type") == "spin") {
            if (meta.find("min") != meta.end()) line << " min " << meta.at("min");
            if (meta.find("max") != meta.end()) line << " max " << meta.at("max");
        } else if (meta.at("type") == "combo") {
            if (meta.find("var") != meta.end()) line << " " << meta.at("var");
        }

        std::cout << line.str() << "\n";
    }

    std::cout << "utttiok\n";
    std::cout.flush();
}

void handle_setoption(QuantumOXEngine& engine, const std::vector<std::string>& tokens) {
    // validate structure: expect 'setoption name <name> value <value>'
    if (tokens.size() < 2) {
        std::cout << "setoption: missing arguments. Type help setoption for more info.\n";
        std::cout.flush();
        return;
    }

    // Validate token keywords and report unknown flags
    std::vector<std::string> valid = VALID_TOKENS.at("setoption");
    for (size_t i = 1; i + 0 < tokens.size(); ++i) {
        std::string t = to_lower_copy(tokens[i]);
        // keywords are "name" and "value"; they should appear where expected
        if (t == "name") {
            // ok
            ++i; // skip the value
            if (i >= tokens.size()) {
                std::cout << "setoption: missing value after 'name'. Type help setoption for more info.\n";
                std::cout.flush();
                return;
            }
            continue;
        } else if (t == "value") {
            ++i;
            if (i >= tokens.size()) {
                std::cout << "setoption: missing value after 'value'. Type help setoption for more info.\n";
                std::cout.flush();
                return;
            }
            continue;
        } else {
            // it's either extra positional value (which is valid) or an unexpected keyword
            // if tokens[i-1] is "name" or "value", it's likely a value, so skip
            if (i >= 1) {
                std::string prev = to_lower_copy(tokens[i-1]);
                if (prev == "name" || prev == "value") {
                    continue; // this is a value, not a flag
                }
            }
            // otherwise treat as unknown flag
            std::string suggestion = suggest_closest(t, valid);
            if (!suggestion.empty()) {
                std::cout << "setoption has no flag \"" << tokens[i] << "\". Did you mean \"" << suggestion << "\"? Type help setoption for more info.\n";
            } else {
                std::cout << "setoption has no flag \"" << tokens[i] << "\". Type help setoption for more info.\n";
            }
            std::cout.flush();
            return;
        }
    }

    auto [name, value] = parse_setoption(tokens);
    if (name.empty()) return;
    auto res = engine.set_option(name, value);
    if (res.first) {
        std::cout << "info string " << res.second << "\n";
    } else {
        std::cout << res.second << "\n";
    }
    std::cout.flush();
}

void handle_grid(QuantumOXEngine& engine, const std::vector<std::string>& tokens) {
    if (tokens.size() < 2) {
        std::cout << "grid: missing subcommand. Type help grid for more info.\n";
        std::cout.flush();
        return;
    }
    std::string sub = tokens[1];
    for (auto & c : sub) c = static_cast<char>(std::tolower(c));

    // valid first-level tokens for grid
    const auto &valid_grid_tokens = VALID_TOKENS.at("grid"); // {"emptygrid", "fill"}

    if (!token_in(sub, valid_grid_tokens)) {
        std::string suggestion = suggest_closest(sub, valid_grid_tokens);
        if (!suggestion.empty()) {
            std::cout << "grid has no subcommand \"" << tokens[1] << "\". Did you mean \"" << suggestion << "\"? Type help grid for more info.\n";
        } else {
            std::cout << "grid has no subcommand \"" << tokens[1] << "\". Type help grid for more info.\n";
        }
        std::cout.flush();
        return;
    }

    if (sub == "emptygrid") {
        engine.new_game();
        if (tokens.size() >= 3) {
            std::string flag = tokens[2];
            std::string lower_flag = to_lower_copy(flag);
            if (lower_flag == "fill") {
                std::vector<int> moves;
                for (size_t i = 3; i < tokens.size(); ++i) {
                    try {
                        int mv = parse_move_token(tokens[i]);
                        moves.push_back(mv);
                    } catch (...) {
                        std::cout << "Invalid move token: " << tokens[i] << "\n";
                        std::cout.flush();
                        return;
                    }
                }
                engine.play_moves(moves);
            } else if (lower_flag == "tttn") {
                if (tokens.size() < 4) {
                    std::cout << "grid emptygrid tttn: missing TTTN string\n";
                    std::cout.flush();
                    return;
                }
                std::string tttn_str = tokens[3];
                // optional: if TTTN might have spaces, concatenate remaining tokens
                for (size_t i = 4; i < tokens.size(); ++i)
                    tttn_str += " " + tokens[i];
            
                try {
                    engine.board.load_tttn(tttn_str);
                } catch (const std::exception &e) {
                    std::cout << "Invalid TTTN: " << e.what() << "\n";
                    std::cout.flush();
                    return;
                }
            } else {
                // unknown flag for emptygrid -- suggest if possible
                std::vector<std::string> sub_flags = {"fill"};
                std::string suggestion = suggest_closest(lower_flag, sub_flags);
                if (!suggestion.empty()) {
                    std::cout << "grid has no flag \"" << flag << "\". Did you mean \"" << suggestion << "\"? Type help grid for more info.\n";
                } else {
                    std::cout << "grid has no flag \"" << flag << "\". Type help grid for more info.\n";
                }
                std::cout.flush();
                return;
            }
        }
    }
}

void handle_go(QuantumOXEngine& engine, const std::vector<std::string>& tokens) {
    std::optional<int> depth = std::nullopt;
    std::optional<int> movetime = std::nullopt;
    std::optional<int> nodes_limit = std::nullopt;

    // recognized flags for go
    const auto &valid_go_flags = VALID_TOKENS.at("go"); // {"depth","movetime","movetime_ms","nodes"}

    for (size_t i = 1; i < tokens.size();) {
        std::string t = tokens[i];
        for (auto & c : t) c = static_cast<char>(std::tolower(c));
        if (t == "depth" && i + 1 < tokens.size()) {
            try { depth = std::stoi(tokens[i+1]); } catch(...) { depth = std::nullopt; }
            i += 2; continue;
        }
        if ((t == "movetime" || t == "movetime_ms") && i + 1 < tokens.size()) {
            try { movetime = std::stoi(tokens[i+1]); } catch(...) { movetime = std::nullopt; }
            i += 2; continue;
        }
        if (t == "nodes" && i + 1 < tokens.size()) {
            try { nodes_limit = std::stoi(tokens[i+1]); } catch(...) { nodes_limit = std::nullopt; }
            i += 2; continue;
        }

        // If we reach here, token t was not recognized as a proper go flag
        std::string suggestion = suggest_closest(t, valid_go_flags);
        if (!suggestion.empty()) {
            std::cout << "go has no flag \"" << tokens[i] << "\". Did you mean \"" << suggestion << "\"? Type help go for more info.\n";
        } else {
            std::cout << "go has no flag \"" << tokens[i] << "\". Type help go for more info.\n";
        }
        std::cout.flush();
        return;
    }

    auto legal = engine.legal_moves();
    if (legal.empty()) {
        std::cout << "bestmove 0000\n";
        std::cout.flush();
        return;
    }

    auto res = engine.go(depth, movetime, nodes_limit);
    for (const auto& line : res.info_lines) {
        std::cout << line << "\n";
        std::cout.flush();
    }
    std::cout << (res.bestmove_line.empty() ? std::string("bestmove 0") : res.bestmove_line) << "\n";
    std::cout.flush();
}

void input_thread_func() {
    std::string raw;
    while (running) {
        if (!std::getline(std::cin, raw)) {
            running = false;
            input_cv.notify_all();
            return;
        }
        {
            std::lock_guard<std::mutex> lk(input_mtx);
            input_queue.push(raw);
        }
        input_cv.notify_one();
    }
}

int main() {
    std::cout << "QuantumOX " << ENGINE_VERSION << " by " << ENGINE_AUTHOR << "\n";
    std::cout.flush();

    QuantumOXEngine engine;

    // async input thread
    std::thread input_thread(input_thread_func);

    while (running) {
        std::unique_lock<std::mutex> lk(input_mtx);

        // wait for a command
        input_cv.wait(lk, [] {
            return !input_queue.empty() || !running;
        });

        if (!running) break;

        std::string raw = input_queue.front();
        input_queue.pop();
        lk.unlock();

        // trim
        size_t start = raw.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = raw.find_last_not_of(" \t\r\n");
        std::string line = raw.substr(start, end - start + 1);
        if (line.empty()) continue;

        std::vector<std::string> tokens = tokenize_command(line);
        if (tokens.empty()) continue;

        std::string cmd = tokens[0];
        for (auto &c : cmd) c = static_cast<char>(std::tolower(c));

        try {
            if (cmd == "uttti") {
                handle_uttti(engine);
            } else if (cmd == "setoption") {
                handle_setoption(engine, tokens);
            } else if (cmd == "isready") {
                std::cout << "readyok\n";
                std::cout.flush();
            } else if (cmd == "utttinewgame") {
                engine.new_game();
            } else if (cmd == "grid") {
                handle_grid(engine, tokens);
            } else if (cmd == "go") {
                if (search_running) {
                    continue;
                }

                search_running = true;
                search_thread = std::thread([&engine, tokens]() {
                    handle_go(engine, tokens);
                    search_running = false;
                });
                search_thread.detach();
            } else if (cmd == "stop") {
                engine.stop();
            } else if (cmd == "clear") {
            #if defined(_WIN32) || defined(_WIN64)
                system("cls");
            #else
                system("clear");
            #endif
            } else if (cmd == "help") {
                if (tokens.size() == 1) {
                    std::cout << HELP_TEXT << "\n";
                } else {
                    std::string topic = tokens[1];
                    for (auto & c : topic) c = static_cast<char>(std::tolower(c));
                    auto it = COMMAND_HELPS.find(topic);
                    if (it != COMMAND_HELPS.end())
                        std::cout << it->second << "\n";
                    else
                        std::cout << "No help for command \"" << tokens[1] << "\".\n";
                }
                std::cout.flush();
            } else if (cmd == "quit" || cmd == "exit") {
                running = false;
                break;
            } else {
                std::cout << "Unknown command: " << tokens[0]
                          << ", type 'help' for UTTTI commands\n";
                std::cout.flush();
            }
        } catch (...) {
            continue;
        }
    }

    running = false;
    input_cv.notify_all();
    if (input_thread.joinable()) input_thread.join();
    return 0;
}
