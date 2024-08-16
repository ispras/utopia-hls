//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_SIMULATOR_H
#define DFCXX_SIMULATOR_H

#include "dfcxx/graph.h"

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace dfcxx {

union SimValue {
  int64_t int_;
  uint64_t uint_;
  double double_;
};

enum class SimType : uint8_t {
  INT = 0,
  UINT,
  FLOAT
};

struct SimVariable {
  std::string name;
  SimType type;

  SimVariable(std::string name, SimType type) : name(name), type(type) {}
};

} // namespace dfcxx

template <>
struct std::hash<dfcxx::SimVariable> {
  size_t operator()(const dfcxx::SimVariable &var) const noexcept {
    return std::hash<std::string>()(var.name);
  }
};

typedef std::unordered_map<dfcxx::SimVariable,
                           std::vector<dfcxx::SimValue>> SimVars;

namespace dfcxx {

class DFCXXSimulator {
public:
  bool simulate(std::ifstream &in, std::ostream &out, std::vector<Node> nodes);
};

} // namespace dfcxx

#endif // DFCXX_SIMULATOR_H
