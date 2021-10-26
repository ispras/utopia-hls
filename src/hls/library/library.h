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

struct VerilogNodeTypePrinter final {
  VerilogNodeTypePrinter(const eda::hls::model::NodeType &rt) : t(rt) {};
  void print(std::ostream &out) const;

  const eda::hls::model::NodeType t;
};

struct VerilogGraphPrinter final {
  VerilogGraphPrinter(const eda::hls::model::Graph &rg) : g(rg) {};

  void printChan(std::ostream &out, const eda::hls::model::Chan &chan) const;
  void print(std::ostream &out) const;

  const eda::hls::model::Graph g;
};

std::ostream& operator <<(std::ostream &out, const VerilogNodeTypePrinter &printer);
std::ostream& operator <<(std::ostream &out, const VerilogGraphPrinter &printer);

} // namespace eda::hls::library
