//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/compiler/compiler.h"

#include "hls/scheduler/dijkstra.h"
#include "utils/assert.h"

#include <filesystem>
#include <fstream>
#include <string>

using Library = eda::hls::library::Library;

namespace fs = std::filesystem;

namespace eda::hls::compiler {

void ExternalModule::addInputs(const std::vector<model::Port*> &inputs) {
  addInput(Port("clock", Port::Direction::IN, Type("clock", 1)));
  addInput(Port("reset", Port::Direction::IN, Type("reset", 1)));
  for (const auto *input : inputs) {
    addInput(Port(utils::replaceSomeChars(input->name),
        Port::Direction::IN, Type("sint", input->type.size)));
  }
}

void ExternalModule::addOutputs(const std::vector<model::Port*> &outputs) {
  for (const auto *output : outputs) {
    addOutput(Port(utils::replaceSomeChars(output->name),
        Port::Direction::OUT, Type("sint", output->type.size)));
  }
}

void FirrtlModule::addInputs(const model::Node *node,
                             const std::vector<model::Chan*> &outputs) {
  for (const auto *output : outputs) {
    addInput(Port::createFirrtlPort(node, output->source.port,
        Port::Direction::IN, Type("sint", output->source.port->type.size)));
  }
}

void FirrtlModule::addOutputs(const model::Node *node,
                              const std::vector<model::Chan*> &inputs) {
  for (const auto *input : inputs) {
    addOutput(Port::createFirrtlPort(node, input->target.port,
        Port::Direction::OUT, Type("sint", input->target.port->type.size)));
  }
}

void Instance::addModuleInputs(const std::vector<model::Port*> &inputs) {
  addModuleInput(Port("clock", Port::Direction::IN, Type("clock", 1)));
  addModuleInput(Port("reset", Port::Direction::IN, Type("reset", 1)));
  for (const auto *input : inputs) {
    addModuleInput(Port(utils::replaceSomeChars(input->name), 
        Port::Direction::IN, Type("sint", input->type.size)));
  }
}

void Instance::addModuleOutputs(const std::vector<model::Port*> &outputs) {
  for (const auto *output : outputs) {
    addModuleOutput(Port(utils::replaceSomeChars(output->name),
        Port::Direction::OUT, Type("sint", output->type.size)));
  }
}

ExternalModule::ExternalModule(const model::NodeType *nodetype) :
    Module(utils::replaceSomeChars(nodetype->name)) {
  addInputs(nodetype->inputs);
  addOutputs(nodetype->outputs);
  /// TODO: Temporal solution.
  auto metaElements = Library::get().find(*nodetype, HWConfig("", "", ""));
  uassert(!metaElements.empty(), "No elements have been found!");
  auto metaElement = metaElements[0];
  //****************************************************************************
  auto element = metaElement->construct();
  setBody(element->ir);
  setPath(element->path);
}

FirrtlModule::FirrtlModule(const eda::hls::model::Model &model,
  const std::string &topModuleName) : Module(topModuleName) {
  const auto* graph = model.findGraph(topModuleName);

  // Inputs & Outputs.
  Port topClockPort = Port("clock", Port::Direction::IN, Type("clock", 1));
  Port topResetPort = Port("reset", Port::Direction::IN, Type("reset", 1));
  addInput(topClockPort);
  addInput(topResetPort);

  uassert(graph, "Top Model graph is null!\n");

  for (const auto *node : graph->nodes) {
    if (node->isSource()) {
      addInputs(node, node->outputs);
    } else if (node->isSink()) {
      addOutputs(node, node->inputs);
    } else {
      Instance instance(utils::replaceSomeChars(node->name),
                        utils::replaceSomeChars(node->type.name));
      Port clockPort = Port(instance.instanceName + "_clock",
                            Port::Direction::IN,
                            Type("clock", 1));
      instance.addInput(clockPort);
      instance.addBinding(topClockPort, clockPort);
      Port resetPort = Port(instance.instanceName + "_reset",
                            Port::Direction::IN,
                            Type("reset", 1));
      instance.addInput(resetPort);
      instance.addBinding(topResetPort, resetPort);
      for (const auto *input : node->inputs) {
        Port inputPort = Port::createFirrtlPort(node, input->target.port,
            Port::Direction::IN, Type("sint", input->source.port->type.size));
        instance.addInput(inputPort);

        if (input->source.node->isSource()) {
          Port connectsFrom = Port::createFirrtlPort(input->source.node,
              input->source.port, Port::Direction::OUT, 
              Type("sint", input->source.port->type.size));
          instance.addBinding(connectsFrom, inputPort);
        }
      }

      instance.addModuleInputs(node->type.inputs);

      for (const auto *output : node->outputs) {
        Port outputPort = Port::createFirrtlPort(node, output->source.port,
            Port::Direction::OUT, Type("sint", output->target.port->type.size));
        instance.addOutput(outputPort);


        if (!node->type.isSink()) {
          Port connectsTo = Port::createFirrtlPort(output->target.node,
              output->target.port, Port::Direction::IN,
              Type("sint", output->target.port->type.size));
          instance.addBinding(outputPort, connectsTo);
        }
      }
      instance.addModuleOutputs(node->type.outputs);
      addInstance(instance);
      setPath("");
    }
  }
}

void ExternalModule::addPrologueToDict(TemplateDictionary *dict) const {
  // Set generation time.
  auto time = std::time(nullptr);
  auto *localTime = std::localtime(&time);
  dict->SetFormattedValue("GEN_TIME",
                          "%d-%d-%d %d:%d:%d",
                          localTime->tm_mday,
                          localTime->tm_mon + 1,
                          localTime->tm_year + 1900,
                          localTime->tm_hour,
                          localTime->tm_min,
                          localTime->tm_sec);

  dict->SetValue("MODULE_NAME", name);
}

void ExternalModule::moveVerilogModule(const std::string &outPath) const {
  fs::path filesystemPath = outPath;
  std::string outputFileName = (((const fs::path) path).filename());
  fs::copy(path, (filesystemPath / outputFileName).string(),
           fs::copy_options::overwrite_existing);
}

void ExternalModule::printVerilogModule(std::ostream &out) const {
  auto *dict = new TemplateDictionary("extModule");
  addPrologueToDict(dict);
  size_t portCount = inputs.size() + outputs.size();
  size_t outputCount = outputs.size();
  addPortsToDict(dict, inputs, "INS", "IN_NAME", "IN_SEP", portCount);
  addPortsToDict(dict, outputs, "OUTS", "OUT_NAME", "OUT_SEP", outputCount);
  dict->SetValue("BODY", body);

  std::string output;
  const char* basePath = std::getenv("UTOPIA_HLS_HOME");
  const char* verilogTemplate = "/src/data/ctemplate/ext_verilog.tpl";
  ctemplate::ExpandTemplate(std::string(basePath) +
                            std::string(verilogTemplate),
                            ctemplate::DO_NOT_STRIP,
                            dict, &output);
  out << output;
}

void ExternalModule::addPortsToDict(TemplateDictionary *dict,
                                    const std::vector<Port> &ports,
                                    const std::string &tagSectName,
                                    const std::string &tagPortName,
                                    const std::string &tagSepName,
                                    const size_t portCount) const {
  for (size_t i = 0; i < ports.size(); i++) {
    const auto &port = ports[i];
    auto *subDict = dict->AddSectionDictionary(tagSectName);
    subDict->SetValue(tagPortName, port.name);
    subDict->SetValue(tagSepName, (i == portCount - 1) ? "" : ",");
  }
}

void FirrtlCircuit::addPortsToDict(TemplateDictionary *dict,
                                   const std::vector<Port> &ports,
                                   const std::string &tagSectName,
                                   const std::string &tagPortName,
                                   const std::string &tagTypeName,
                                   const std::string &tagSepName,
                                   const size_t portCount) const {
  for (size_t i = 0; i < ports.size(); i++) {
    const auto &port = ports[i];
    auto *subDict = dict->AddSectionDictionary(tagSectName);
    subDict->SetValue(tagPortName, port.name);
    const auto type = port.type;

    std::string portTypeName = type.name;
    portTypeName = portTypeName + type.toString();
    subDict->SetValue(tagTypeName, portTypeName);
    subDict->SetValue(tagSepName, (i == portCount - 1) ? "" : ",");
  }
}

void FirrtlCircuit::addInstancesToDict(TemplateDictionary *dict,
    const std::vector<Instance> &instances) const {
  for (const auto &instance : instances) {
    auto *instDict = dict->AddSectionDictionary("INSTS");
    size_t portCount = instance.inputs.size() + instance.outputs.size();
    size_t outputCount = instance.moduleOutputs.size();
    for (size_t i = 0; i < instance.inputs.size(); i++) {
      const auto input = instance.inputs[i];
      auto *inInstDict = instDict->AddSectionDictionary("INST_INS");
      inInstDict->SetValue("INST_IN_NAME", input.name);
      inInstDict->SetValue("INST_IN_SEP", (i == portCount - 1) ? "" : ",");
    }
    for (size_t i = 0; i < instance.outputs.size(); i++) {
      const auto output = instance.outputs[i];
      auto *outInstDict = instDict->AddSectionDictionary("INST_OUTS");
      outInstDict->SetValue("INST_OUT_NAME", output.name);
      outInstDict->SetValue("INST_OUT_SEP", (i == outputCount - 1) ? "" : ",");
    }
    instDict->SetValue("INSTANCE_NAME", instance.instanceName);
    instDict->SetValue("MODULE_NAME", instance.moduleName);
    addPortsToDict(instDict, instance.moduleInputs, "MODULE_INS", "MOD_IN_NAME",
        "MOD_IN_TYPE", "MOD_IN_SEP", portCount);
    addPortsToDict(instDict, instance.moduleOutputs, "MODULE_OUTS",
        "MOD_OUT_NAME", "MOD_OUT_TYPE", "MOD_OUT_SEP", outputCount);
    // Connections.
    for (const auto &pair : instance.bindings) {
      auto *conDict = dict->AddSectionDictionary("CONS");
      const auto input = pair.first;
      const auto output = pair.second;
      conDict->SetValue("CON_IN_NAME", input.name);
      conDict->SetValue("CON_OUT_NAME", output.name);
      std::string inTypeName = input.type.name;
      inTypeName = inTypeName + input.type.toString();
      conDict->SetValue("CON_IN_TYPE", inTypeName);
      std::string outTypeName = output.type.name;
      outTypeName = outTypeName + output.type.toString();
      conDict->SetValue("CON_OUT_TYPE", outTypeName);
    }
  }
}

void FirrtlCircuit::addExtModulesToDict(TemplateDictionary *dict,
    const std::map<std::string, ExternalModule> &extModules) const {
  for (const auto &pair : extModules) {
    const auto extModule = pair.second;
    auto *extDict = dict->AddSectionDictionary("EXTS");
    size_t portCount = extModule.inputs.size() + extModule.outputs.size();
    size_t outputCount = extModule.outputs.size();
    extDict->SetValue("EXTMODULE_NAME", extModule.name);
    addPortsToDict(extDict, extModule.inputs, "EXT_INS", "EXT_IN_NAME",
        "EXT_IN_TYPE", "EXT_IN_SEP", portCount);
    addPortsToDict(extDict, extModule.outputs, "EXT_OUTS", "EXT_OUT_NAME",
        "EXT_OUT_TYPE", "EXT_OUT_SEP", outputCount);
  }
}

void FirrtlCircuit::addPrologueToDict(TemplateDictionary *dict) const {
  // Set generation time.
  auto time = std::time(nullptr);
  auto *localTime = std::localtime(&time);
  dict->SetFormattedValue("GEN_TIME",
                          "%d-%d-%d %d:%d:%d",
                          localTime->tm_mday,
                          localTime->tm_mon + 1,
                          localTime->tm_year + 1900,
                          localTime->tm_hour,
                          localTime->tm_min,
                          localTime->tm_sec);

  dict->SetValue("CIRCUIT_NAME", name);
}

void FirrtlCircuit::printFirrtl(std::ostream &out) const {
  auto *dict = new TemplateDictionary("top");
  addPrologueToDict(dict);

  for (const auto &pair : firModules) {
    const auto firModule = pair.second;
    size_t portCount = firModule.inputs.size() + firModule.outputs.size();
    size_t outputCount = firModule.outputs.size();
    // Inputs.
    addPortsToDict(dict, firModule.inputs, "TOP_INS", "TOP_IN_NAME",
        "TOP_IN_TYPE", "TOP_IN_SEP", portCount);
    // Outputs.
    addPortsToDict(dict, firModule.outputs, "TOP_OUTS", "TOP_OUT_NAME",
        "TOP_OUT_TYPE", "TOP_OUT_SEP", outputCount);
    // Instances.
    addInstancesToDict(dict, firModule.instances);
  }

  // External modules declaration.
  addExtModulesToDict(dict, extModules);
  // Use template to store result to file.
  std::string output;
  const char *basePath = std::getenv("UTOPIA_HLS_HOME");
  const char *verilogTemplate = "/src/data/ctemplate/top_firrtl.tpl";
  ctemplate::ExpandTemplate(std::string(basePath) + 
                            std::string(verilogTemplate),
                            ctemplate::DO_NOT_STRIP,
                            dict,
                            &output);
  out << output;
}

void FirrtlCircuit::dumpVerilogLibrary(const std::string &outPath,
                                       std::ostream      &out) const {
  for (const auto &pair : extModules) {
    if (pair.second.body == "") {
      pair.second.moveVerilogModule(outPath);
    } else {
      pair.second.printVerilogModule(out);
    }
  }
}

FirrtlCircuit::FirrtlCircuit(const Model &model, const std::string &name) :
    name(name), latency(model.ind.ticks), resetInitialValue(0) {
  for (auto nodeTypeIterator = model.nodetypes.cbegin();
       nodeTypeIterator != model.nodetypes.cend();
       nodeTypeIterator++) {
    addExternalModule(nodeTypeIterator->second);
  }
  addFirrtlModule(FirrtlModule(model, name));
};

void FirrtlCircuit::printFiles(const std::string &outFirFileName,
                               const std::string &outVlogLibName,
                               const std::string &outVlogTopName,
                               const std::string &outPath) const {
  // Create path.
  fs::path fsOutPath = outPath;
  fs::create_directories(fsOutPath);

  // Create FIRRTL file.
  std::ofstream outputFile;
  outputFile.open((fsOutPath / outFirFileName).string());
  uassert(outputFile, "Can't open outputFile!\n");
  printFirrtl(outputFile);
  outputFile.close();

  // Lower FIRRTL to verilog and move result to output directory.
  dumpVerilogOptFile((fsOutPath / outFirFileName).string());
  uassert(fs::exists(name + ".sv"),
          "File main.sv is not found! (CIRCT doesn't work)\n");
  fs::rename(name + ".sv", ((fsOutPath / outVlogTopName).string()).c_str());

  // Get Verilog code for library elements and move it to output directory.
  outputFile.open((fsOutPath / outVlogLibName).string());
  dumpVerilogLibrary(fsOutPath.string(), outputFile);
  outputFile.close();
}

void FirrtlCircuit::printRndVlogTest(const Model &model,
                                     const std::string &outPath,
                                     const std::string &outTestFileName,
                                     const size_t testCount) const {
  const fs::path fsPath = outPath;
  fs::create_directories(fsPath);

  std::ofstream testBenchFile;
  testBenchFile.open(fsPath / outTestFileName);

  TemplateDictionary *dict = new TemplateDictionary("tb");

  // Set generation time.
  auto time = std::time(nullptr);
  auto *localTime = std::localtime(&time);
  dict->SetFormattedValue("GEN_TIME",
                          "%d-%d-%d %d:%d:%d",
                          localTime->tm_mday,
                          localTime->tm_mon + 1,
                          localTime->tm_year + 1900,
                          localTime->tm_hour,
                          localTime->tm_min,
                          localTime->tm_sec);

  const auto &main = (firModules.find(name))->second;

  dict->SetValue("MODULE_NAME", main.name);

  // Set registers for device inputs.
  std::vector<Port> inputs = main.inputs;
  std::vector<std::string> bndNames;
  for (const auto &input : inputs) {
    TemplateDictionary *inDict = dict->AddSectionDictionary("INS");
    if (input.isClock() || input.isReset()) {
      inDict->SetValue("IN_TYPE", "");
    } else {
      inDict->SetValue("IN_TYPE", 
          "[" + std::to_string(input.type.width - 1) + ":0]");
    }

    const std::string iName = utils::replaceSomeChars(input.name);
    inDict->SetValue("IN_NAME", iName);
    bndNames.push_back(iName);
  }

  // Set registers for device outputs.
  std::vector<Port> outputs = main.outputs;

  for (const auto &output : outputs) {
    TemplateDictionary *outDict = dict->AddSectionDictionary("OUTS");
    outDict->SetValue("OUT_TYPE", 
        "[" + std::to_string(output.type.width - 1) + ":0]");
    const std::string oName = utils::replaceSomeChars(output.name);
    outDict->SetValue("OUT_NAME", oName);
    bndNames.push_back(oName);
  }

  // Calculate model's latency and store it.
  dict->SetIntValue("LATENCY", latency);

  // Set bindings to device under test.
  for (size_t i = 0; i < bndNames.size(); i++) {
    TemplateDictionary *bndDict = dict->AddSectionDictionary("BIND");
    bndDict->SetValue("WNAME", bndNames[i]);
    bndDict->SetValue("SEP", (i == bndNames.size() - 1) ? "" : ",\n");
  }
  bndNames.clear();

  // Set resets.
  for (const auto &input : inputs) {
    if (input.isReset()) {
      auto *resetDict = dict->AddSectionDictionary("RESETS");
      resetDict->SetValue("RESET_NAME", input.name);
      resetDict->SetIntValue("RESET_VALUE", resetInitialValue);
    }
  }

  // Generate random stimuli.
  for (size_t i = 0; i < testCount; i++) {
    TemplateDictionary *tDict = dict->AddSectionDictionary("TESTS");
    for (const auto &input : inputs) {

      if (input.isClock() || input.isReset())
        continue;

      TemplateDictionary *sDict = tDict->AddSectionDictionary("ST");
      sDict->SetValue("NAME", utils::replaceSomeChars(input.name));
    }
  }

  // Use the template to store testbench to file.
  std::string output;
  const char *basePath = std::getenv("UTOPIA_HLS_HOME");
  const char *verilogTemplate = "/src/data/ctemplate/tbench_verilog.tpl";
  ctemplate::ExpandTemplate(std::string(basePath) + 
                            std::string(verilogTemplate),
                            ctemplate::DO_NOT_STRIP, dict, &output);
  testBenchFile << output;
  testBenchFile.close();
}

} // namespace eda::hls::compiler