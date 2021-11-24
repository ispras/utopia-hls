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
  struct Chan;
  struct Graph;
  struct Model;
  struct Node;
  struct NodeType;
  struct Port;
} // namespace eda::hls::model

using namespace eda::hls::model;

namespace eda::hls::parser::dfc {

class Builder final: public eda::util::Singleton<Builder> {
  friend class eda::util::Singleton<Builder>;

public:
  std::shared_ptr<Model> create(const std::string &name);

  void startKernel(const std::string &name);

  void declareWire(const ::dfc::wire *wire);

  void connectWires(const ::dfc::wire *in, const ::dfc::wire *out);

  void connectWires(const std::string &opcode,
                    const std::vector<const ::dfc::wire*> &in,
                    const std::vector<const ::dfc::wire*> &out);

private:
  Builder() {}

  struct Unit final {
    Unit(const std::string &opcode,
         const std::vector<const ::dfc::wire*> &in,
         const std::vector<const ::dfc::wire*> &out):
      opcode(opcode), in(in), out(out) {}

    std::string fullName() const;

    /// Unit operation.
    std::string opcode;
    /// Unique inputs.
    std::vector<const ::dfc::wire*> in;
    /// Unique outputs.
    std::vector<const ::dfc::wire*> out;
  };

  struct Kernel final {
    Kernel(const std::string &name): name(name) {}

    /// Kernel name.
    std::string name;
    /// Contains all units.
    std::vector<Unit> units;
    /// Contains all wires.
    std::vector<const ::dfc::wire*> wires;
    /// Contains the wires connected to the given point (another wire).
    std::unordered_map<std::string, std::vector<const ::dfc::wire*>> fanout;
  };

  Port* createPort(const ::dfc::wire *wire, unsigned latency);
  NodeType* createNodetype(const Unit &unit, Model *model);
  Chan* createChan(const ::dfc::wire *wire, Graph *graph);
  Node* createNode(const Unit &unit, Graph *graph, Model *model);
  Graph* createGraph(const Kernel &kernel, Model *model);

  std::vector<Kernel> kernels;
};

} // namespace eda::hls::parser::dfc
