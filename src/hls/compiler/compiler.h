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
#include "util/string.h"

#include <ctemplate/template.h>

namespace eda::hls::compiler {

struct Type final {
  const std::string name;
  const std::size_t width;

  Type(const std::string &name,
       const std::size_t width) :
       name(name), width(width) {};

  std::string toString() const {
    return (((this->name != "clock") && (this->name != "reset")) ?
                  ("<" + std::to_string(this->width) + ">") : 
                  "");
  }
};

struct Port final {
  enum Direction { IN, OUT, INOUT };
  const std::string name;
  const Direction dir;
  const Type type;

  Port(const std::string &name,
       const Direction dir,
       const Type &type) :
    name(name), dir(dir), type(type) {}

  static Port createFirrtlPort(const model::Node *node,
                               const model::Port *port,
                               const Port::Direction dir,
                               const Type type) {
    return Port(replaceSomeChars(node->name) +
                  "_" +
                  replaceSomeChars(port->name),
                dir, type);
  }

  bool isClock() const {
    return type.name == "clock";
  }

  bool isReset() const {
    return type.name == "reset";
  }
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

  void addModuleInput(const Port &moduleInPort) {
    moduleInputs.push_back(moduleInPort);
  }

  void addOutput(const Port &outputPort) {
    outputs.push_back(outputPort);
  }

  void addModuleOutput(const Port &moduleOutPort) {
    moduleOutputs.push_back(moduleOutPort);
  }

  void addBinding(const Port &connectsFrom, const Port &connectsTo) {
    bindings.push_back({connectsFrom, connectsTo});
  }

  void addModuleInputs(const std::vector<model::Port*> &inputs);
  void addModuleOutputs(const std::vector<model::Port*> &outputs);
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
  FirrtlModule(const model::Model &model, const std::string &topName);
  void printFirrtlModule(std::ostream &out) const;
  void addInstance(const Instance &inInstance) {
    instances.push_back(inInstance);
  }
private:
  void addInputs(const model::Node *node,
                 const std::vector<model::Chan*> &outputs);
  void addOutputs(const model::Node *node,
                  const std::vector<model::Chan*> &inputs);
};

struct ExternalModule final : Module {
  ExternalModule(const model::NodeType *nodetype);
  void printVerilogModule(std::ostream &out) const;
  void moveVerilogModule(const std::string &outPath) const;
private:
  void addInputs(const std::vector<model::Port*>  &inputs);
  void addOutputs(const std::vector<model::Port*> &outputs);
  void addPortsToDict(ctemplate::TemplateDictionary *dict,
                      const std::vector<Port> &ports,
                      const std::string &tagSectName,
                      const std::string &tagPortName,
                      const std::string &tagSepName,
                      const size_t portCount) const;
  void addPrologueToDict(ctemplate::TemplateDictionary *dict) const;
};

class FirrtlCircuit final {
private:
  const std::string name;
  const int latency;
  int resetInitialValue;
  std::map<std::string, FirrtlModule> firModules;
  std::map<std::string, ExternalModule> extModules;
  void addPortsToDict(ctemplate::TemplateDictionary *dict,
                      const std::vector<Port> &ports,
                      const std::string &tagSectName,
                      const std::string &tagPortName,
                      const std::string &tagTypeName,
                      const std::string &tagSepName,
                      const size_t portCount) const;
  void addPrologueToDict(ctemplate::TemplateDictionary *dict) const;
  void addInstancesToDict(ctemplate::TemplateDictionary *dict,
                          const std::vector<Instance> &instances) const;
  void addExtModulesToDict(ctemplate::TemplateDictionary *dict,
      const std::map<std::string, ExternalModule> &extModules) const;
  void dumpVerilogOptFile(const std::string &inFirName) const {
    system((std::string(FirrtlCircuit::circt) +
           inFirName +
           std::string(FirrtlCircuit::circtOptions)).c_str());
  }
  void dumpVerilogLibrary(const std::string &outPath,
                          std::ostream &out) const;
  void printFirrtl(std::ostream &out) const;
  void addFirrtlModule(const FirrtlModule &firModule) {
    firModules.insert({firModule.getName(), firModule});
  }
  void addExternalModule(const ExternalModule &extModule) {
    extModules.insert({extModule.getName(), extModule});
  }
public:
  static constexpr const char* indent        = "    ";
  static constexpr const char* opPrefix      = "firrtl.";
  static constexpr const char* typePrefix    = "!firrtl.";
  static constexpr const char* varPrefix     = "%";
  static constexpr const char* circt         = "circt-opt ";
  static constexpr const char* circtOptions  = " --lower-firrtl-to-hw \
                                                 --export-split-verilog";
  FirrtlCircuit(const Model &model, const std::string &name);
  /**
   * @brief Constructs output files and moves them to the given directory.
   *
   * @param outFirFileName Name of output FIRRTL file.
   * @param outVlogLibName Name of output file that contains library modules.
   * @param outVlogTopName Name of top verilog file.
   * @param outPath        Path to store output files.
   *
   * @returns Nothing, but output files should be created.
   */
  void printFiles(const std::string& outFirFileName,
                  const std::string& outVlogLibName,
                  const std::string& outVlogTopName,
                  const std::string& outPath) const;
  /**
   * @brief Generates a Verilog random testbench for the input model.
   *
   * @param model           Input model.
   * @param outPath         Path to store output testbench.
   * @param outTestFileName Filename for output testbench.
   * @param testCount       Number of test stimuli at random sequence.
   *
   * @returns Nothing, but Verilog testbench should be created.
   */
  void printRndVlogTest(const Model       &model,
                        const std::string &outPath,
                        const std::string &outTestFileName,
                        const size_t       testCount) const;

};

struct Compiler final {

  Compiler() = default;

  /**
   * @brief Generates a FIRRTL IR for the input model.
   *
   * @param model Input model.
   * @param name  Name of top module in FIRRTL circuit to be constructed.
   * 
   * @returns Constructed FIRRTL circuit.
   */
  std::shared_ptr<FirrtlCircuit> constructFirrtlCircuit(const Model &model,
      const std::string &name = "main") const {
    return std::make_shared<FirrtlCircuit>(model, name);
   }
};

//std::ostream& operator <<(std::ostream &out, const FirrtlCircuit &firCircuit);

} // namespace eda::hls::compiler
