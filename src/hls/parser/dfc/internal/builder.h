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

#include <cassert>
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

  /// Creates a model that contains all defined kernels (graphs).
  std::shared_ptr<Model> create(const std::string &modelName);

  /// Creates a model that contains a kernel (graph) w/ the given name.
  std::shared_ptr<Model> create(const std::string &modelName,
                                const std::string &kernelName);

  void startKernel(const std::string &name);

  void declareWire(const ::dfc::wire *wire);

  void connectWires(const ::dfc::wire *in, const ::dfc::wire *out);

  void connectWires(const std::string &opcode,
                    const std::vector<const ::dfc::wire*> &in,
                    const std::vector<const ::dfc::wire*> &out);

private:
  Builder(): common("<common>") {}

  struct Unit;

  struct Wire final {
    Wire(const std::string &name,
         const eda::hls::model::Type &type,
         bool isInput,
         bool isOutput,
         bool isConst,
         const std::string &value):
      name(name),
      type(type),
      isInput(isInput),
      isOutput(isOutput),
      isConst(isConst),
      value(value) {}

    void setConsumer(Unit *unit) {
      assert(!consumer && "Multiple reads");
      consumer = unit;
    }

    void setProducer(Unit *unit) {
      assert(!producer && "Multiple writes");
      producer = unit;
    }

    const std::string name;
    const eda::hls::model::Type type;

    const bool isInput;
    const bool isOutput;
    const bool isConst;

    const std::string value;

    Unit *producer = nullptr;
    Unit *consumer = nullptr;
  };

  struct Unit final {
    Unit(const std::string &opcode,
         const std::vector<Wire*> &in,
         const std::vector<Wire*> &out):
        opcode(opcode), in(in), out(out) {
      for (auto *wire : in) {
        wire->setConsumer(this);
      }
      for (auto *wire : out) {
        wire->setProducer(this);
      }
    }

    std::string getFullName() const;

    void addInput(Wire *wire) {
      in.push_back(wire);
      wire->setConsumer(this);
    }

    void addOutput(Wire *wire) {
      out.push_back(wire);
      wire->setProducer(this);
    }

    Signature getSignature() const {
      std::vector<std::string> inputTypeNames;
      std::vector<std::string> outputTypeNames;
      for (const auto *input : in) {
        inputTypeNames.push_back(input->type.name);
      }
      for (const auto *output : out) {
        outputTypeNames.push_back(output->type.name);
      }
      return Signature(getFullName(),
                       inputTypeNames,
                       outputTypeNames);
    }

    const std::string opcode;
    std::vector<Wire*> in;
    std::vector<Wire*> out;
  };

  struct Kernel final {
    Kernel(const std::string &name): name(name) {}

    enum Mode {
      CREATE_ORIGINAL,
      CREATE_VERSION,
      ACCESS_ORIGINAL,
      ACCESS_VERSION
    };

    Wire* getWire(const ::dfc::wire *wire, Mode mode);

    Unit* getUnit(const std::string &opcode,
                  const std::vector<Wire*> &in,
                  const std::vector<Wire*> &out);

    void connect(Wire *source, Wire *target);

    void transform();

    const std::string name;

    std::vector<Wire*> wires;
    std::vector<Unit*> units;

    /// Maps name to the related wire.
    std::unordered_map<std::string, Wire*> originals;
    /// Maps name to the latest version of the wire.
    std::unordered_map<std::string, Wire*> versions;

    bool isTransformed = false;
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

  static Graph* getGraph(const Kernel *kernel, Model *model);

  Kernel* getKernel() {
    return kernels.empty() ? &common : kernels.back();
  }

  /// Common part of all kernels (usually, wires and constants).
  Kernel common;
  /// Kernels.
  std::vector<Kernel*> kernels;
};

} // namespace eda::hls::parser::dfc
