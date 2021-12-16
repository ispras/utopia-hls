//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

#include <iostream>

using namespace eda::hls::model;

namespace eda::hls::compiler {

struct Wire final {
  std::string name;
  size_t width;
  std::string type;

  Wire(const std::string &name,
       const size_t width = 1,
       const std::string &type = "uint") :
    name(name),
    width(width),
    type(type) {}
};


struct Port final {
  std::string name;
  bool isInput;
  size_t width;
  std::string type;

  Port(const std::string &name,
       const bool isInput,
       const size_t width = 1,
       const std::string &type = "uint") :
    name(name),
    isInput(isInput),
    width(width),
    type(type) {}
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

  void addInput(const Port &inputPort);
  void addOutput(const Port &outputPort);
  void addModuleInput(const Port &moduleInputPort);
  void addModuleOutput(const Port &moduleOutputPort);
  void addBinding(const Port &connectsFrom, const Port &connectTo);
};


struct Module {

  std::string moduleName;
  std::vector<Wire> wires;
  std::vector<Instance> instances;
  std::vector<Port> inputs;
  std::vector<Port> outputs;
  std::string body;

  Module() = default;

  void addBody(const std::string &body);
  void addInput(const Port &inputPort);
  void addOutput(const Port &outputPort);
  void printBody(std::ostream &out) const;
  void printEmptyLine(std::ostream &out) const;

};


struct FirrtlModule final : Module {

  FirrtlModule(const Model &model);

  //void addWire(const Wire &inputWire);
  void addInstance(const Instance &inputInstance);
  //void printWires(std::ostream &out) const;
  void printInstances(std::ostream &out) const;
  void printConnections(std::ostream &out) const;
  void printDeclaration(std::ostream &out) const;
  void printEpilogue(std::ostream &out) const;;
  void printFirrtl(std::ostream &out) const;

};


struct ExternalModule final : Module {

  ExternalModule(const model::NodeType *nodetype);

  void printDeclaration(std::ostream &out) const;
  void printFirrtlDeclaration(std::ostream &out) const;
  void printEpilogue(std::ostream &out) const;
  void printVerilog(std::ostream &out) const;

};


struct Circuit final {
  std::string name;
  std::map<std::string, FirrtlModule> firModules;
  std::map<std::string, ExternalModule> externalModules;

  Circuit(std::string moduleName);

  void addFirModule(const FirrtlModule &firModule);
  void addExternalModule(const ExternalModule &externalModule);
  void convertToSV(const std::string& inputFirrtlName) const;
  void print(std::ostream &out) const;
  void printFirrtl(std::ostream &out) const;
  void printVerilog(std::ostream &out) const;
  void printFiles(const std::string &outputFirrtlName,
                  const std::string &outputVerilogName,
                  const std::string &outputDirectoryName) const;
};

struct Compiler final {
  static constexpr const char* indent = "    ";
  static constexpr const char* opPrefix = "firrtl.";
  static constexpr const char* typePrefix = "!firrtl.";
  static constexpr const char* varPrefix = "%";
  static constexpr const char* circt = "circt-opt ";
  static constexpr const char* circt_options = " --lower-firrtl-to-hw \
                                                 --export-split-verilog";

  const std::shared_ptr<Model> model;

  Compiler(const Model &model);

  void print(std::ostream &out) const;

  std::shared_ptr<Circuit> constructCircuit();
};

std::ostream& operator <<(std::ostream &out, const Circuit &circuit);

} // namespace eda::hls::compiler
