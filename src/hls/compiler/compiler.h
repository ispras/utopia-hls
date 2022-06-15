//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
#pragma once
#include "hls/library/library.h"

#include "hls/model/model.h"
#include "hls/scheduler/dijkstra.h"

#include <ctemplate/template.h>

#include <iostream>
#include <memory>

namespace eda::hls::compiler {

struct Type final {
  std::string name;
  std::size_t element_width;

  Type(const std::string &name,
       const std::size_t element_width) :
       name(name),
       element_width(element_width) {};
};

struct Port final {
  enum Direction {
    IN,
    OUT,
    INOUT
  };
  std::string name;
  Direction direction;
  Type type;

  Port(const std::string &name,
       const Direction direction,
       const Type &type) :
    name(name),
    direction(direction),
    type(type) {}

  bool isClock() const;
};

struct Instance final {
  std::string instanceName;
  std::string moduleName;
  std::vector<Port> inputs;
  std::vector<Port> outputs;
  std::vector<Port> moduleInputs;
  std::vector<Port> moduleOutputs;
  std::vector<std::pair<Port, Port>> bindings;

  Instance(const std::string &instanceName,
           const std::string &moduleName) :
    instanceName(instanceName),
    moduleName(moduleName),
    inputs(),
    outputs(),
    moduleInputs(),
    moduleOutputs(),
    bindings() {}

  void addInput(const Port &inputPort) {
    inputs.push_back(inputPort);
  }

  void addModuleInput(const Port &moduleInputPort) {
    moduleInputs.push_back(moduleInputPort);
  }

  void addModuleOutput(const Port &moduleOutputPort) {
    moduleOutputs.push_back(moduleOutputPort);
  }

  void addOutput(const Port &outputPort) {
    inputs.push_back(outputPort);
  }

  void addBinding(const Port &connectsFrom, const Port &connectTo) {
    bindings.push_back({connectsFrom, connectTo});
  }
  void addModuleInputs(const std::vector<model::Port*> inputs);
  void addModuleOutputs(const std::vector<model::Port*> outputs);
};

struct Module {
  std::string name;
  std::vector<Port> inputs;
  std::vector<Port> outputs;
  std::string path;
  std::string body;

  Module(const std::string &name) : name(name) {};

  void addInput(const Port &inputPort) {
    inputs.push_back(inputPort);
  }

  void addOutput(const Port &outputPort) {
    outputs.push_back(outputPort);
  }

  void addPath(const std::string &path) {
    this->path = path;
  }

  void addBody(const std::string &body) {
    this->body = body;
  }

  void printBody(std::ostream &out) const {
    out << body;
  }

  std::string getName() const {
    return name;
  };
};

struct FirrtlModule final : Module {
  FirrtlModule(const model::Model &model, const std::string &topModuleName);
  void printFirrtlModule(std::ostream &out) const;
  void addInstance(const Instance &inputInstance) {
    instances.push_back(inputInstance);
  }
private:
  std::vector<Instance> instances;

  void addInputs(const model::Node *node,
                 const std::vector<model::Chan*> outputs);
  void addOutputs(const model::Node *node,
                  const std::vector<model::Chan*> inputs);
  void printDeclaration(std::ostream &out) const;
  void printInstances(std::ostream &out) const;
  void printConnections(std::ostream &out) const;
  void printEpilogue(std::ostream &out) const;
};

struct ExternalModule final : Module {
  ExternalModule(const model::NodeType *nodetype);
  void printVerilogModule(std::ostream &out) const;
  void moveVerilogModule(const std::string &outputDirName) const;
  void printFirrtlDeclaration(std::ostream &out) const;
private:
  void addInputs(const std::vector<model::Port*> inputs);
  void addOutputs(const std::vector<model::Port*> outputs);
  void printEpilogue(std::ostream &out) const;
  void printDeclaration(std::ostream &out) const;
};

class Circuit final {
private:
  std::string name;
  std::map<std::string, FirrtlModule> firModules;
  std::map<std::string, ExternalModule> extModules;
  void printDeclaration(std::ostream &out) const;
  void printFirrtlModules(std::ostream &out) const;
  void printExtDeclarations(std::ostream &out) const;
  void printEpilogue(std::ostream &out) const;
public:
  static constexpr const char* indent = "    ";
  static constexpr const char* opPrefix = "firrtl.";
  static constexpr const char* typePrefix = "!firrtl.";
  static constexpr const char* varPrefix = "%";
  static constexpr const char* circt = "circt-opt ";
  static constexpr const char* circt_options = " --lower-firrtl-to-hw \
                                                 --export-split-verilog";
  Circuit(const std::string& name) : name(name) {};
  void printFiles(const std::string& outputFirrtlName,
                  const std::string& outputVerilogName,
                  const std::string& outputDirName) const;
  void convertToSV(const std::string& inputFirrtlName) const;

  void moveVerilogLibrary(const std::string &outputDirName,
                          std::ostream &out) const;
  void printFirrtl(std::ostream &out) const;
  void printFirrtlModule(const FirrtlModule &firmodule,
                         std::ostream &out) const;


  void addFirModule(const FirrtlModule &firModule) {
    firModules.insert({firModule.getName(), firModule});
  }

  void addExternalModule(const ExternalModule &externalModule) {
    extModules.insert({externalModule.getName(), externalModule});
  }

  /**
   * @brief Generates a Verilog random testbench for the current model.
   *
   * @param tstPath Path to testbench file to be created.
   * @param tstCnt Number of test stimuli at random sequence
   * @return Nothing, but "*.v" testbench should be created.
   */
  void printRndVlogTest(const model::Model &model,
                        const std::string &tstPath,
                        const int latency,
                        const size_t tstCnt);

};

struct Compiler final {

  Compiler() = default;

  /**
   * @brief Generates a FIRRTL IR for the input model.
   *
   * @param Input model.
   * @param Name of the top module to be generated.
   * @return Constructed circuit.
   */
  std::shared_ptr<Circuit> constructCircuit(const model::Model &model,
                                            const std::string &name);
};

std::ostream& operator <<(std::ostream &out, const Circuit &circuit);

} // namespace eda::hls::compiler
