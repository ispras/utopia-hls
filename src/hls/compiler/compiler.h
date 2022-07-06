//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "hls/model/model.h"

#include <ctemplate/template.h>

namespace eda::hls::compiler {

struct Type final {
  const std::string name;
  const std::size_t width;

  Type(const std::string &name,
       const std::size_t width) :
       name(name), width(width) {};
};

struct Port final {
  enum Direction { IN, OUT, INOUT };
  const std::string name;
  const Direction direction;
  const Type type;

  Port(const std::string &name,
       const Direction direction,
       const Type &type) :
    name(name), direction(direction), type(type) {}

  bool isClock() const;
};

struct Instance final {
  const std::string instanceName;
  const std::string moduleName;
  std::vector<Port> inputs;
  std::vector<Port> outputs;
  std::vector<Port> moduleInputs;
  std::vector<Port> moduleOutputs;
  std::vector<std::pair<Port, Port>> bindings;

  Instance(const std::string &instanceName,
           const std::string &moduleName) :
    instanceName(instanceName), moduleName(moduleName),
    inputs(), outputs(), moduleInputs(), moduleOutputs(), bindings() {}

  void addInput(const Port &inputPort) {
    inputs.push_back(inputPort);
  }

  void addModuleInput(const Port &moduleInputPort) {
    moduleInputs.push_back(moduleInputPort);
  }

  void addOutput(const Port &outputPort) {
    outputs.push_back(outputPort);
  }

  void addModuleOutput(const Port &moduleOutputPort) {
    moduleOutputs.push_back(moduleOutputPort);
  }

  void addBinding(const Port &connectsFrom, const Port &connectTo) {
    bindings.push_back({connectsFrom, connectTo});
  }

  void addModuleInputs(const std::vector<model::Port*> inputs);
  void addModuleOutputs(const std::vector<model::Port*> outputs);
};

struct Module {
  const std::string name;
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

  void setPath(const std::string &path) {
    this->path = path;
  }

  void setBody(const std::string &body) {
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
  std::vector<Instance> instances;
  FirrtlModule(const model::Model &model, const std::string &topModuleName);
  void printFirrtlModule(std::ostream &out) const;
  void addInstance(const Instance &inputInstance) {
    instances.push_back(inputInstance);
  }
private:
  void addInputs(const model::Node *node,
                 const std::vector<model::Chan*> outputs);
  void addOutputs(const model::Node *node,
                  const std::vector<model::Chan*> inputs);
};

struct ExternalModule final : Module {
  ExternalModule(const model::NodeType *nodetype);
  void printVerilogModule(std::ostream &out) const;
  void moveVerilogModule(const std::string &outputDirName) const;
  //void printFirrtlDeclaration(std::ostream &out) const;
private:
  void addInputs(const std::vector<model::Port*> inputs);
  void addOutputs(const std::vector<model::Port*> outputs);
  void printEpilogue(std::ostream &out) const;
  void printDeclaration(std::ostream &out) const;
  static std::string getFileNameFromPath(const std::string &path);
  void addPortsToDict(ctemplate::TemplateDictionary *dict,
                      const std::vector<Port> ports,
                      const std::string &tagSectName,
                      const std::string &tagPortName,
                      const std::string &tagSepName,
                      const size_t portCount) const;
  void addPrologueToDict(ctemplate::TemplateDictionary *dict) const;
};

struct FirrtlCircuit final {
private:
  std::string name;
  std::map<std::string, FirrtlModule> firModules;
  std::map<std::string, ExternalModule> extModules;
  void addPortsToDict(ctemplate::TemplateDictionary *dict,
                      const std::vector<Port> ports,
                      const std::string &tagSectName,
                      const std::string &tagPortName,
                      const std::string &tagTypeName,
                      const std::string &tagSepName,
                      const size_t portCount) const;
  void addPrologueToDict(ctemplate::TemplateDictionary *dict) const;
  void addInstancesToDict(ctemplate::TemplateDictionary *dict,
                          std::vector<Instance> instances) const;
  void addExtModulesToDict(ctemplate::TemplateDictionary *dict,
      std::map<std::string, ExternalModule> extModules) const;
public:
  static constexpr const char* indent        = "    ";
  static constexpr const char* opPrefix      = "firrtl.";
  static constexpr const char* typePrefix    = "!firrtl.";
  static constexpr const char* varPrefix     = "%";
  static constexpr const char* circt         = "circt-opt ";
  static constexpr const char* circtOptions = " --lower-firrtl-to-hw \
                                                --export-split-verilog";
  FirrtlCircuit(const std::string& name) : name(name) {};

  void printFiles(const std::string& outputFirrtlName,
                  const std::string& outputVerilogLibraryName,
                  const std::string& outputVerilogTopModuleName,
                  const std::string& outputDirName) const;
  void dumpVerilogOptFile(const std::string& inputFirrtlName) const;

  void dumpVerilogLibrary(const std::string &outputDirName,
                          std::ostream &out) const;
  void printFirrtl(std::ostream &out) const;

  void addFirModule(const FirrtlModule &firModule) {
    firModules.insert({firModule.getName(), firModule});
  }

  void addExternalModule(const ExternalModule &externalModule) {
    extModules.insert({externalModule.getName(), externalModule});
  }

  /**
   * @brief Generates a Verilog random testbench for the current model
   *
   * @param model Reference to model
   * @param outPath Output path
   * @param outTestFileName Output testbench
   * @param latency Latency
   * @param tstCnt Number of test stimuli at random sequence
   * @return Nothing, but Verilog testbench should be created
   */
   void printRndVlogTest(const Model &model,
                         const std::string &outPath,
                         const std::string &outTestFileName,
                         const int latency,
                         const size_t tstCnt);

};

struct Compiler final {

  Compiler() = default;

  /**
   * @brief Generates a FIRRTL IR for the input model
   *
   * @param Input model
   * @param Name of the top module to be generated
   * @return Constructed circuit
   */
  std::shared_ptr<FirrtlCircuit> constructCircuit(const model::Model &model,
                                                  const std::string &name);
};

std::ostream& operator <<(std::ostream &out, const FirrtlCircuit &circuit);

} // namespace eda::hls::compiler
