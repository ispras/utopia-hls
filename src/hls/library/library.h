//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <iostream>

#include "hls/model/model.h"

namespace eda::hls::library {

/* comment for possible future usage
struct Port {
  enum Direction { IN, OUT, INOUT };

  Port(const std::string &name, const Direction direction, const unsigned width) :
    name(name), direction(direction), width(width) {};

  std::string name;
  Direction direction;
  unsigned width;
}; */

struct VerilogPrinter {
  VerilogPrinter(eda::hls::model::NodeType &type) : type(type) {};

  eda::hls::model::NodeType type;
};

std::ostream& operator <<(std::ostream &out, const VerilogPrinter &printer);

} // namespace eda::hls::library
