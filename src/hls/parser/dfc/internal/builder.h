//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/parser/dfc/stream.h"
#include "util/singleton.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace eda::hls::model {
  struct Model;
} // namespace eda::hls::model

namespace eda::hls::parser::dfc {

class Builder final: public eda::util::Singleton<Builder> {
public:
  std::shared_ptr<eda::hls::model::Model> create(const std::string &name);

  void startKernel(const std::string &name) {
    // TODO:
  }

  void declareWire(const ::dfc::wire *wire) {
    wires.push_back(wire);
  }

  void connectWires(const ::dfc::wire *in, const ::dfc::wire *out) {
    fanout[in->name].push_back(out);
  }

  void connectWires(const std::string &opcode,
                    const std::vector<const ::dfc::wire*> &in,
                    const std::vector<const ::dfc::wire*> &out) {
    // Create new inputs and connect them w/ the old ones.
    std::vector<const ::dfc::wire*> new_in(in.size());
    for (const auto *source : in) {
      const auto *target = source->new_wire();
      connectWires(source, target);
      new_in.push_back(target);
    }

    // Create a unit w/ the newly created inputs.
    Unit unit(opcode, new_in, out);
    units.push_back(std::move(unit));
  }

private:
  struct Unit {
    Unit(const std::string &opcode,
         const std::vector<const ::dfc::wire*> &in,
         const std::vector<const ::dfc::wire*> &out):
      opcode(opcode), in(in), out(out) {}

    std::string opcode;
    std::vector<const ::dfc::wire*> in;
    std::vector<const ::dfc::wire*> out;
  };

  std::vector<Unit> units;
  std::vector<const ::dfc::wire*> wires;
  std::unordered_map<std::string, std::vector<const ::dfc::wire*>> fanout;
};

} // namespace eda::hls::parser::dfc
