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
         const std::vector<std::string> &in,
         const std::vector<std::string> &out):
      opcode(opcode), in(in), out(out) {}

    std::string fullName() const;

    /// Unit operation.
    const std::string opcode;
    /// Unique inputs.
    std::vector<std::string> in;
    /// Unique outputs.
    std::vector<std::string> out;
  };

  struct Kernel final {
    Kernel(const std::string &name): name(name) {}

    enum Mode {
      CREATE_ORIGINAL,
      CREATE_VERSION,
      ACCESS_ORIGINAL,
      ACCESS_VERSION
    };

    Wire* getWire(const std::string &name) const;
    Wire* getWire(const ::dfc::wire *wire, Mode mode);

    void reduce();

    /// Kernel name.
    const std::string name;

    /// Contains all units.
    std::vector<Unit*> units;
    /// Contains all wires.
    std::vector<Wire*> wires;

    /// Maps name to the related wire.
    std::unordered_map<std::string, Wire*> originals;
    /// Maps name to the latest version of the wire.
    std::unordered_map<std::string, Wire*> versions;

    /// Maps wire to the one it is replaced with.
    std::unordered_map<std::string, Wire*> replaced;

    /// Maps non-input wire to its source.
    std::unordered_map<std::string, std::string> in;
    /// Maps non-output wire to its targets.
    std::unordered_map<std::string, std::vector<std::string>> out;
  };

  static Port* getPort(const Wire *wire, unsigned latency);

  static Chan* getChan(const Wire *wire, Graph *graph);

  static NodeType* getNodetype(const Kernel *kernel,
                               const Unit *unit,
                               Model *model);

  static Node* getNode(const Kernel *kernel,
                       const Unit *unit,
                       Graph *graph,
                       Model *model);

  static Graph* getGraph(const Kernel *kernel,
                         Model *model);

  std::vector<Kernel*> kernels;
};

} // namespace eda::hls::parser::dfc
