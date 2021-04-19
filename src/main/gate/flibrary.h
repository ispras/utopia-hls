/*
 * Copyright 2021 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#pragma once

#include <memory>
#include <vector>

#include "rtl/fsymbol.h"

using namespace eda::rtl;

namespace eda {
namespace gate {

class Netlist;

/**
 * \brief Interface for functional library.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>.
 */
struct FLibrary {
  typedef std::vector<unsigned> Arg;
  typedef Arg Out;
  typedef std::vector<Arg> In;

  /// Checks if the library support the given function.
  virtual bool supports(FuncSymbol func) const = 0;

  /// Synthesize the netlist for the given function.
  virtual bool synthesize(FuncSymbol func, const Out &out, const In &in, Netlist &net) = 0;

  virtual ~FLibrary() {} 
};

class FLibraryDefault final: public FLibrary {
public:
  static FLibrary& instance() {
    if (_instance == nullptr) {
      _instance = std::unique_ptr<FLibrary>(new FLibraryDefault());
    }
    return *_instance;
  }

  bool supports(FuncSymbol func) const override;
  bool synthesize(FuncSymbol func, const Out &out, const In &in, Netlist &net) override;

private:
  FLibraryDefault() {}
  ~FLibraryDefault() override {}

  bool synthesize_nop(const Out &out, const In &in, Netlist &net);
  bool synthesize_not(const Out &out, const In &in, Netlist &net);
  bool synthesize_and(const Out &out, const In &in, Netlist &net);
  bool synthesize_add(const Out &out, const In &in, Netlist &net);
  bool synthesize_sub(const Out &out, const In &in, Netlist &net);
  bool synthesize_mux(const Out &out, const In &in, Netlist &net);

  static std::unique_ptr<FLibrary> _instance;
};

}} // namespace eda::gate

