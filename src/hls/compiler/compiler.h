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

namespace eda::hls::compiler {

struct Wire final {
  std::string wireName;
  size_t wireWidth;
  std::string wireType;

  Wire(const std::string &wireName,
       const size_t wireWidth = 1,
       const std::string &wireType = "uint") :
   wireName(wireName),
   wireWidth(wireWidth),
   wireType(wireType) {}
};


struct Port final {
  std::string portName;
  bool inputPort;
  size_t portWidth;
  std::string portType;

  Port(const std::string &portName,
       const bool inputPort,
       const size_t portWidth = 1,
       const std::string &portType = "uint") :
    portName(portName),
    inputPort(inputPort),
    portWidth(portWidth),
    portType(portType) {}
};


struct Instance final {
  std::string instanceName;
  std::string moduleName;
  std::vector<Port> inputs;
  std::vector<Port> outputs;
  std::vector<std::pair<Port, Port>> bindings;

  Instance(const std::string &instanceName,
           const std::string &moduleName) :
            instanceName(instanceName),
            moduleName(moduleName),
            inputs(),
            outputs(),
            bindings() {}

  void addInput(const Port &inputPort);
  void addOutput(const Port &outputPort);
  void addBinding(const Port &connectsFrom, const Port &connectTo);
};


struct Module final {
  std::string moduleName;
  std::vector<Wire> wires;
  std::vector<Instance> instances;
  std::vector<Port> inputs;
  std::vector<Port> outputs;
  std::string body;

  Module(const eda::hls::model::Model &model);

  void addBody(const std::string &body);
  void addWire(const Wire &inputWire);
  void addInstance(const Instance &inputInstance);
  void addPort(const Port &port, const bool isInput);
  void printWires(std::ostream &out) const;
  void printInstances(std::ostream &out) const;
  void printConnections(std::ostream &out) const;
  void printDeclaration(std::ostream &out) const;
  void printBody(std::ostream &out) const;
  void printEpilogue(std::ostream &out) const;
  void printEmptyLine(std::ostream &out) const;
  void printFirrtl(std::ostream &out) const;
};


struct ExternalModule final {
  std::string externalModuleName;
  std::vector<Wire> wires;
  std::vector<Instance> instances;
  std::vector<Port> inputs;
  std::vector<Port> outputs;
  std::string body;

  ExternalModule(const model::NodeType *nodetype);

  void addBody(const std::string &body);
  void addPort(const Port &port, const bool isInput);
  void printDeclaration(std::ostream &out) const;
  void printFirrtlDeclaration(std::ostream &out) const;
  void printBody(std::ostream &out) const;
  void printEpilogue(std::ostream &out) const;;
  void printEmptyLine(std::ostream &out) const;
  void printVerilog(std::ostream &out) const;
};


struct Circuit final {
  std::string circuitName;
  std::map<std::string, Module> modules;
  std::map<std::string, ExternalModule> externalModules;

  Circuit(std::string moduleName);

  void addModule(const Module &module);
  void addExternalModule(const ExternalModule &externalModule);
  void print(std::ostream &out) const;
  void printFirrtl(std::ostream &out) const;
  void printVerilog(std::ostream &out) const;
  void printFiles(const std::string& outputFirrtlName,
                  const std::string& outputVerilogName) const;
};

struct Compiler final {
  static constexpr const char* indent = "    ";
  static constexpr const char* opPrefix = "firrtl.";
  static constexpr const char* typePrefix = "!firrtl.";
  static constexpr const char* varPrefix = "%";

  const eda::hls::model::Model model;

  Compiler(const eda::hls::model::Model &model);

  void print(std::ostream &out) const;

  std::shared_ptr<Circuit> constructCircuit();
};

std::ostream& operator <<(std::ostream &out, const Circuit &circuit);

} // namespace eda::hls::compiler
