//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>
#include <string>

namespace eda::hls::library {

typedef struct Port {
  typedef enum { IN, OUT, INOUT } Direction;

  Port(const std::string &name, const Direction direction, const unsigned width) :
    name(name), direction(direction), width(width) {};

  std::string name;
  Direction direction;
  unsigned width;
} Port;

struct ModuleInterface {
  ModuleInterface(const std::string &name) : name(name) {};

  void addPort(Port &port) {
    interface.push_back(port);
  };

  std::string name;
  std::vector<Port> interface;
};

} // namespace eda::hls::library
