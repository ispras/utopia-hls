//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/compiler/compiler.h"
#include "hls/debugger/debugger.h"

#include "hls/library/library.h"

#include "hls/model/model.h"
#include "hls/scheduler/dijkstra.h"

#include <ctemplate/template.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string.h>

using namespace eda::hls::model;

namespace eda::hls::compiler {

struct Type final {
  std::string name;
  std::size_t element_width;
  bool isContainer;
  std::size_t element_count;

  Type(const std::string &name,
       const std::size_t element_width,
       bool isContainer,
       const std::size_t element_count) :
       name(name),
       element_width(element_width),
       isContainer(isContainer),
       element_count(element_count) {}
};

struct Port final {
  std::string name;
  bool isInput;
  Type type;

  Port(const std::string &name,
       const bool isInput,
       const Type &type) :
    name(name),
    isInput(isInput),
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

  void addInput(const Port &inputPort);
  void addOutput(const Port &outputPort);
  void addModuleInput(const Port &moduleInputPort);
  void addModuleOutput(const Port &moduleOutputPort);
  void addBinding(const Port &connectsFrom, const Port &connectTo);
};

struct Module {
  std::string name;
  std::vector<Instance> instances;
  std::vector<Port> inputs;
  std::vector<Port> outputs;
  std::string path;
  std::string body;

  Module(const std::string &name) : name(name) {};

  void addPath(const std::string &path);
  void addBody(const std::string &body);
  void addInput(const Port &inputPort);
  void addOutput(const Port &outputPort);
};

struct FirrtlModule final : Module {
  FirrtlModule(const model::Model &model, const std::string &topModuleName);
  //void addWire(const Wire &inputWire);
  void addInstance(const Instance &inputInstance);
};

struct ExternalModule final : Module {
  ExternalModule(const model::NodeType *nodetype);
};

struct Circuit final {
  std::string name;
  std::map<std::string, FirrtlModule> firModules;
  std::map<std::string, ExternalModule> extModules;

  Circuit(const std::string &moduleName);

  void addFirModule(const FirrtlModule &firModule);
  void addExternalModule(const ExternalModule &externalModule);
  FirrtlModule* findFirModule(const std::string &name) const;
  ExternalModule* findExtModule(const std::string &name) const;
  Module* findModule(const std::string &name) const;
  Module* findMain() const;
};

struct Compiler final {
  static constexpr const char* indent = "    ";
  static constexpr const char* opPrefix = "firrtl.";
  static constexpr const char* typePrefix = "!firrtl.";
  static constexpr const char* varPrefix = "%";
  static constexpr const char* circt = "circt-opt ";
  static constexpr const char* circt_options = " --lower-firrtl-to-hw \
                                                 --export-split-verilog";

  const std::shared_ptr<model::Model> model;
  std::shared_ptr<Circuit> circuit;

  Compiler(const model::Model &model);

  std::shared_ptr<Circuit> constructCircuit(const std::string& topModuleName);

  void printBody(const Module &module, std::ostream &out) const;
  void printEmptyLine(std::ostream &out) const;

  //void printWires(std::ostream &out) const;
  void printInstances(const FirrtlModule &firmodule, std::ostream &out) const;
  void printConnections(const FirrtlModule &firmodule, std::ostream &out) const;
  void printDeclaration(const FirrtlModule &firmodule, std::ostream &out) const;
  void printEpilogue(const FirrtlModule &firmodule, std::ostream &out) const;
  void printFirrtlModule(const FirrtlModule &firmodule,
                         std::ostream &out) const;

  void printDeclaration(const ExternalModule &extmodule,
                        std::ostream &out) const;
  void printFirrtlDeclaration(const ExternalModule &extmodule,
                              std::ostream &out) const;
  void printEpilogue(const ExternalModule &extmodule, std::ostream &out) const;
  void moveVerilogLibrary(const std::string &outputDirectoryName,
                          std::ostream &out) const;
  void moveVerilogModule(const std::string &outputDirectoryName,
                         const ExternalModule &extModule) const;
  void printVerilogModule(const std::string &outputDirectoryName,
                          const ExternalModule &extmodule,
                          std::ostream &out) const;

  void convertToSV(const std::string& inputFirrtlName) const;
  void printFirrtl(std::ostream &out) const;
  void printFiles(const std::string &outputFirrtlName,
                  const std::string &outputVerilogName,
                  const std::string &outputDirectoryName) const;

  /**
   * @brief Generates a Verilog random testbench for the current model.
   *
   * @param tstPath Path to testbench file to be created.
   * @param tstCnt Number of test stimuli at random sequence
   * @return Nothing, but Verilog testbench should be created.
   */
  void printRndVlogTest(const std::string &tstPath, const int tstCnt);
};

std::ostream& operator <<(std::ostream &out, const Circuit &circuit);

} // namespace eda::hls::compiler
