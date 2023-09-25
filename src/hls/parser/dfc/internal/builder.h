//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/parser/dfc/stream.h"
#include "utils/assert.h"
#include "utils/singleton.h"

#include <cassert>
#include <memory>
#include <set>
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

using Chan = eda::hls::model::Chan;
using Graph = eda::hls::model::Graph;
using Model = eda::hls::model::Model;
using Node = eda::hls::model::Node;
using NodeType = eda::hls::model::NodeType;
using Port = eda::hls::model::Port;
using Signature = eda::hls::model::Signature;

namespace eda::hls::parser::dfc {

class Builder final: public eda::utils::Singleton<Builder> {
  friend class eda::utils::Singleton<Builder>;

public:

  /// Creates a model that contains all defined kernels (graphs).
  std::shared_ptr<Model> create(const std::string &modelName);

  /// Creates a model that contains a kernel (graph) w/ the given name.
  std::shared_ptr<Model> create(const std::string &modelName,
                                const std::string &kernelName);

  void startKernel(const std::string &name);

  void activateKernel() {
    activeKernels.push_back(kernels.back());
  }

  void deactivateKernel() {
    Kernel *kernel = activeKernels.back();
    kernel->transform();
    activeKernels.pop_back();
  }

  struct Kernel;

  Kernel *getKernel() {
    return activeKernels.empty() ? &common : activeKernels.back();
  }

  Kernel *getKernel(const std::string &name) {
    auto i = std::find_if(kernels.begin(), kernels.end(),
        [&name](Kernel *kernel) { return kernel->name == name; });
    assert(i != kernels.end() && "Kernel does not exist");
    auto *kernel = *i;
    return kernel;
  }

  void declareWire(const ::dfc::wire *wire);

  void connectWires(const ::dfc::wire *in,
                    const ::dfc::wire *out);

  void connectToInstanceInput(const std::string &instanceName,
                              const ::dfc::wire *wire,
                              const std::string &inputName);

  void connectToInstanceOutput(const std::string &instanceName,
                               const ::dfc::wire *wire,
                               const std::string &outputName);

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

  struct Instance;

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

  Unit(const std::string &opcode,
       const std::string &instanceName,
       const std::string &kernelName):
      opcode(opcode) {
      instance = new Instance(instanceName, kernelName);
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

    /// Unit can be an instance of some other kernel.
    Instance *instance;
  };

  struct Instance final {
    const std::string instanceName;
    const std::string kernelName;
    std::unordered_map<Wire*, std::string> bindingsToInputs;
    std::unordered_map<Wire*, std::string> bindingsToOutputs;
    std::unordered_map<std::string, Unit*> sourcesWaitConnection;
    std::unordered_map<std::string, Unit*> sinksWaitConnection;

    public:
    Instance(const std::string &instanceName,
             const std::string &kernelName) : instanceName(instanceName),
                                              kernelName(kernelName) {
      auto *kernel = Builder::get().getKernel(kernelName);
      for (const auto &[wire_name, source] : kernel->inputNamesToSources) {
        sourcesWaitConnection.insert({wire_name, source});
      }
      for (const auto &[wire_name, sink] : kernel->outputNamesToSinks) {
        sinksWaitConnection.insert({wire_name, sink});
      }
    };

    void addBindingToInput(Wire *wire, const std::string &inputName) {
      auto *source = sourcesWaitConnection.find(inputName)->second;
      uassert(source != nullptr,
              "Input with such name doesn't exist!\n");
      uassert(source->out.front()->type == wire->type,
              "Types of connecting wires doesn't match!\n");
      sourcesWaitConnection.erase(inputName);
      bindingsToInputs.insert({wire, inputName});
    }

    void addBindingToOutput(Wire *wire, const std::string &outputName) {
      auto *sink = sinksWaitConnection.find(outputName)->second;
      uassert(sink != nullptr,
              "Input with such name doesn't exist!\n");
      uassert(sink->in.front()->type == wire->type,
              "Types of connecting wires doesn't match!\n");
      sinksWaitConnection.erase(outputName);
      bindingsToOutputs.insert({wire, outputName});
    }

    void modifyBindingToInput(Wire *oldWire,
                              Wire *newWire,
                              const std::string &inputName) {
      bindingsToInputs.erase(oldWire);
      bindingsToInputs.insert({newWire, inputName});
    }

    bool isFullyConnected() {
      return ((sourcesWaitConnection.size() == 0) &&
              (sinksWaitConnection.size()   == 0));
    }

  };

  public:
  struct Kernel final {
    Kernel(const std::string &name): name(name) {}

    enum Mode {
      CREATE_ORIGINAL,
      CREATE_VERSION,
      ACCESS_ORIGINAL,
      ACCESS_VERSION
    };

    Wire *getWire(const ::dfc::wire *wire, Mode mode);

    Unit *getUnit(const std::string &opcode,
                  const std::vector<Wire*> &in,
                  const std::vector<Wire*> &out);

    void createInstanceUnit(const std::string &opcode,
                            const std::string &instanceName,
                            const std::string &kernelName);

    void connect(Wire *source, Wire *target);

    void transform();

    const std::string name;

    std::vector<Wire*> wires;
    std::vector<Unit*> units;

    /// Sources and sinks needed to connect to the kernel.
    std::unordered_map<std::string, Unit*> inputNamesToSources;
    std::unordered_map<std::string, Unit*> outputNamesToSinks;

    /// Kernel instances.
    std::map<std::string, Unit*> instanceUnits;

    /// Maps name to the related wire.
    std::unordered_map<std::string, Wire*> originals;
    /// Maps name to the latest version of the wire.
    std::unordered_map<std::string, Wire*> versions;



    bool isTransformed = false;
  };
  private:

  static std::string getSourceName(const Unit *source);

  static std::string getSinkName(const Unit *sink);

  static Port *getPort(const Wire *wire, const unsigned latency);

  static Chan *getChan(const Wire *wire, Graph *graph);

  static NodeType *getNodetype(const Unit *unit,
                               std::shared_ptr<Model> model);

  static Node *getNode(const Kernel *kernel,
                       const Unit *unit,
                       Graph *graph,
                       std::shared_ptr<Model> model);

  static Graph *getGraph(const Kernel *kernel, std::shared_ptr<Model> model);

  /// Common part of all kernels (usually, wires and constants).
  Kernel common;
  /// Kernels.
  std::vector<Kernel*> kernels;
  /// Active kernels.
  std::vector<Kernel*> activeKernels;
};

} // namespace eda::hls::parser::dfc