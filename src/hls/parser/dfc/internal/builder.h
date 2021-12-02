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

  struct Wire final {
    Wire(const std::string &name, const std::string &type, bool input, bool output):
      name(name), type(type), input(input), output(output) {}

    const std::string name;
    const std::string type;
    const bool input;
    const bool output;
  };

  struct Unit final {
    Unit(const std::string &opcode,
         const std::vector<Wire*> &in,
         const std::vector<Wire*> &out):
      opcode(opcode), in(in), out(out) {}

    std::string fullName() const;

    /// Unit operation.
    const std::string opcode;
    /// Unique inputs.
    std::vector<Wire*> in;
    /// Unique outputs.
    std::vector<Wire*> out;
  };

  struct Kernel final {
    Kernel(const std::string &name): name(name) {}

    enum Mode { CREATE, ACCESS, CREATE_COPY, CREATE_IF_NOT_EXISTS };
    Wire* getWire(const std::string &name) const;
    Wire* getWire(const ::dfc::wire *wire, Mode mode);

    /// Kernel name.
    const std::string name;
    /// Contains all units.
    std::vector<Unit*> units;
    /// Contains all wires.
    std::unordered_map<std::string, Wire*> wires;
    /// Maps a non-input wire to its source.
    std::unordered_map<std::string, Wire*> in;
    /// Maps a non-output wire to its targets.
    std::unordered_map<std::string, std::vector<Wire*>> out;
  };

  Port* createPort(const Wire *wire, unsigned latency);
  NodeType* createNodetype(const Unit *unit, Model *model);
  Chan* createChan(const Wire *wire, Graph *graph);
  Node* createNode(const Unit *unit, Graph *graph, Model *model);
  Graph* createGraph(const Kernel *kernel, Model *model);

  std::vector<Kernel*> kernels;
};

} // namespace eda::hls::parser::dfc
