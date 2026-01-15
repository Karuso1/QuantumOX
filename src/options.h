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

#ifndef OPTIONS_H
#define OPTIONS_H

#include <string>
#include <unordered_map>
#include <functional>
#include <vector>
#include "constants.h"

namespace QuantumOX {

    // --- Option class -----------------------------------------------------------
    class Option {
    public:
        std::string name;
        std::string type; // "string", "bool", etc.
        std::string default_value;
        std::string value;
        std::string description;
        std::function<std::string(const std::string&)> validator;
    
        Option(const std::string& n, const std::string& t, const std::string& def,
               const std::string& desc = "",
               std::function<std::string(const std::string&)> val = nullptr);
        
        void set(const std::string& raw_value);
    };
    
    // --- validators -------------------------------------------------------------
    std::string validate_grid(const std::string& v);
    std::string validate_firstplayer(const std::string& v);
    std::string validate_threads(const std::string& v);
    std::string validate_hash(const std::string& v);
    
    // --- options registry -------------------------------------------------------
    extern std::unordered_map<std::string, Option> _registry;
    
    // --- public API -------------------------------------------------------------
    std::pair<bool, std::string> set_option(const std::string& name, const std::string& raw_value);
    std::string get_option(const std::string& name);
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> list_options();
    std::vector<int> get_grid_dims();

} // namespace QuantumOX

#endif // OPTIONS_H
