
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

#include "options.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <thread>     // for hardware_concurrency
#include <cstdlib>    // for std::stoi

namespace QuantumOX {

    // --- Option class definitions ----------------------------------------------
    Option::Option(const std::string& n, const std::string& t, const std::string& def,
                   const std::string& desc,
                   std::function<std::string(const std::string&)> val)
        : name(n), type(t), default_value(def), value(def), description(desc), validator(val) {}
    
    void Option::set(const std::string& raw_value) {
        std::string v;
    
        if (type == "bool") {
            std::string sv = raw_value;
            std::transform(sv.begin(), sv.end(), sv.begin(), ::tolower);
            if (sv == "true" || sv == "1" || sv == "yes" || sv == "on") {
                v = "true";
            } else if (sv == "false" || sv == "0" || sv == "no" || sv == "off") {
                v = "false";
            } else {
                throw std::runtime_error("Option " + name + " expects a bool-like value");
            }
        } else {
            v = raw_value;
        }
    
        if (validator) v = validator(v);
    
        value = v;
    }
    
    // --- validators -------------------------------------------------------------
    std::string validate_grid(const std::string& v) {
        if (std::find(SUPPORTED_GRIDS.begin(), SUPPORTED_GRIDS.end(), v) == SUPPORTED_GRIDS.end()) {
            std::ostringstream oss;
            oss << "Unsupported grid '" << v << "'. Supported: ";
            for (size_t i = 0; i < SUPPORTED_GRIDS.size(); ++i) {
                oss << SUPPORTED_GRIDS[i];
                if (i != SUPPORTED_GRIDS.size() - 1) oss << ", ";
            }
            throw std::runtime_error(oss.str());
        }
    
        // verify parse_grid_spec doesn't throw
        parse_grid_spec(v);
        return v;
    }
    
    std::string validate_firstplayer(const std::string& v) {
        std::string sv = v;
        std::transform(sv.begin(), sv.end(), sv.begin(), ::toupper);
        if (sv != std::string(1, SYMBOL_X) && sv != std::string(1, SYMBOL_O)) {
            throw std::runtime_error("FirstPlayer must be 'X' or 'O'");
        }
        return sv;
    }

    // Validate Threads: must be integer between 1 and 512 (inclusive)
    std::string validate_threads(const std::string& v) {
        int n = 0;
        try {
            size_t idx = 0;
            n = std::stoi(v, &idx);
            if (idx != v.size()) {
                throw std::invalid_argument("extra characters");
            }
        } catch (const std::exception&) {
            throw std::runtime_error("Threads expects an integer value");
        }

        if (n < 1 || n > 512) {
            std::ostringstream oss;
            oss << "Threads must be between 1 and 512 (requested " << n << ")";
            throw std::runtime_error(oss.str());
        }
        return std::to_string(n);
    }

    // Validate Hash: must be integer between 1 and 2097152
    std::string validate_hash(const std::string& v) {
        int n = 0;
        try {
            size_t idx = 0;
            n = std::stoi(v, &idx);
            if (idx != v.size()) {
                throw std::invalid_argument("extra characters");
            }
        } catch (const std::exception&) {
            throw std::runtime_error("Hash expects an integer value");
        }

        if (n < 1 || n > 2097152) {
            std::ostringstream oss;
            oss << "Hash must be between 1 and 2097152 (requested " << n << ")";
            throw std::runtime_error(oss.str());
        }
        return std::to_string(n);
    }
    
    // --- default registry -------------------------------------------------------
    std::unordered_map<std::string, Option> _registry = {
        {"Grid", Option("Grid", "combo", DEFAULT_GRID,
                        "Board grid specification, e.g. '3x3', '4x4', or '3x3x3'.",
                        validate_grid)},
        {"FirstPlayer", Option("FirstPlayer", "combo", std::string(1, SYMBOL_X),
                           "Symbol for the player who moves first: 'X' or 'O'.",
                           validate_firstplayer)},
        {"Hash", Option("Hash", "spin", std::to_string(DEFAULT_HASH),
                        "Number of hash used to store positions (1-2097152).",
                        validate_hash)},
        {"Threads", Option("Threads", "spin", std::to_string(DEFAULT_THREADS),
                           "Number of worker threads used for search (1-512).",
                           validate_threads)}
    };
    
    // --- public API -------------------------------------------------------------
    std::pair<bool, std::string> set_option(const std::string& name, const std::string& raw_value) {
        auto it = _registry.find(name);
        if (it == _registry.end()) {
            return {false, "Unknown option '" + name + "'"};
        }
    
        try {
            it->second.set(raw_value);
        } catch (const std::exception& e) {
            return {false, "Failed to set option '" + name + "': " + e.what()};
        }
    
        return {true, "set \"" + name + "\" to " + it->second.value};
    }
    
    std::string get_option(const std::string& name) {
        auto it = _registry.find(name);
        if (it == _registry.end()) {
            throw std::runtime_error("Unknown option '" + name + "'");
        }
        return it->second.value;
    }
    
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> list_options() {
        std::unordered_map<std::string, std::unordered_map<std::string, std::string>> out;
        for (const auto& [k, opt] : _registry) {
            std::unordered_map<std::string, std::string> meta;
        
            // Handle combo type options for UTTTI
            if (opt.name == "Grid") {
                meta["type"] = "combo";
                meta["default"] = opt.default_value;
                std::ostringstream vars;
                for (const auto& g : SUPPORTED_GRIDS) {
                    if (!vars.str().empty()) vars << " ";
                    vars << "var " << g;
                }
                meta["var"] = vars.str();
            } else if (opt.name == "FirstPlayer") {
                meta["type"] = "combo";
                meta["default"] = opt.default_value;
                meta["var"] = "var X var O";
            } else if (opt.name == "Hash") {
                meta["type"] = "spin";
                meta["default"] = opt.default_value;
                meta["min"] = "1";
                meta["max"] = "2097152";
                std::ostringstream varss;
                varss << "min 1 max 2097152";
                meta["var"] = varss.str();
            } else if (opt.name == "Threads") {
                meta["type"] = "spin";
                meta["default"] = opt.default_value;
                // expose a friendly var/min/max for UIs
                meta["min"] = "1";
                meta["max"] = "512";
                std::ostringstream varss;
                varss << "min 1 max 512";
                meta["var"] = varss.str();
            } else {
                meta["type"] = opt.type;
                meta["default"] = opt.default_value;
            }
        
            meta["value"] = opt.value;
            meta["description"] = opt.description;
            out[k] = std::move(meta);
        }
        return out;
    }
    
    std::vector<int> get_grid_dims() {
        std::string grid_spec = get_option("Grid");
        return parse_grid_spec(grid_spec);
    }

} // namespace QuantumOX
