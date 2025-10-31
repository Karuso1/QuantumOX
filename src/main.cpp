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

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <utility>

#include "constants.h"
#include "utils.h"
#include "options.h"
#include "engine.h"

using namespace QuantumOX;

static const char* HELP_TEXT = R"(
QuantumOX UTTTI commands:
  uttti                     - handshake
  setoption name <name> value <value>  - set engine option
  isready                   - check if engine is ready
  utttinewgame              - reset board silently
  grid emptygrid [fill ...] - reset board and optionally fill moves
  go depth <N>              - run search to depth N
  quit / exit               - exit engine
  help                      - show this help text
)";

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
    if (tokens.size() < 2) return;
    std::string sub = tokens[1];
    for (auto & c : sub) c = static_cast<char>(std::tolower(c));
    if (sub == "emptygrid") {
        engine.new_game();
        if (tokens.size() >= 4 && std::string(tokens[2]) == "fill") {
            std::vector<int> moves;
            for (size_t i = 3; i < tokens.size(); ++i) {
                try {
                    int mv = parse_move_token(tokens[i]);
                    moves.push_back(mv);
                } catch (...) {
                    return;
                }
            }
            engine.play_moves(moves);
        }
    }
}

void handle_go(QuantumOXEngine& engine, const std::vector<std::string>& tokens) {
    std::optional<int> depth = std::nullopt;
    std::optional<int> movetime = std::nullopt;
    std::optional<int> nodes_limit = std::nullopt;
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
        ++i;
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

int main() {
    std::cout << "QuantumOX " << ENGINE_VERSION << " by " << ENGINE_AUTHOR << "\n";
    std::cout.flush();
    
    QuantumOXEngine engine;
    std::string raw;
    while (std::getline(std::cin, raw)) {
        std::string line;
        // trim
        size_t start = raw.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = raw.find_last_not_of(" \t\r\n");
        line = raw.substr(start, end - start + 1);
        if (line.empty()) continue;
        std::vector<std::string> tokens = tokenize_command(line);
        if (tokens.empty()) continue;
        std::string cmd = tokens[0];
        for (auto & c : cmd) c = static_cast<char>(std::tolower(c));

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
                handle_go(engine, tokens);
            } else if (cmd == "stop") {
                engine.stop();
            } else if (cmd == "help") {
                std::cout << HELP_TEXT << "\n";
                std::cout.flush();
            } else if (cmd == "quit" || cmd == "exit") {
                break;
            } else {
                std::cout << "Unknown command: " << tokens[0] << ", type 'help' for UTTTI commands\n";
                std::cout.flush();
            }
        } catch (...) {
            // ignore errors and continue loop
            continue;
        }
    }
    return 0;
}
