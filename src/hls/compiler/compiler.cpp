//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/compiler/compiler.h"
#include "hls/debugger/debugger.h"
#include "hls/library/library.h"
#include "hls/library/library_mock.h"
#include "hls/scheduler/dijkstra.h"
#include "hls/scheduler/scheduler.h"

#include <ctemplate/template.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string.h>

using namespace eda::hls::debugger;
using namespace eda::hls::library;
using namespace eda::hls::model;
using namespace eda::hls::scheduler;

namespace eda::hls::compiler {

const std::string chanSourceToString(const eda::hls::model::Chan &chan) {
  return chan.source.node->name + "_" + chan.source.port->name;
}

bool Port::isClock() const {
  return name == "clock";
}

void Instance::addInput(const Port &inputPort) {
  inputs.push_back(inputPort);
}

void Instance::addModuleInput(const Port &moduleInputPort) {
  moduleInputs.push_back(moduleInputPort);
}

void Instance::addModuleOutput(const Port &moduleOutputPort) {
  moduleOutputs.push_back(moduleOutputPort);
}

void Instance::addOutput(const Port &outputPort) {
  inputs.push_back(outputPort);
}

void Instance::addBinding(const Port &connectsFrom, const Port &connectTo) {
  bindings.push_back({connectsFrom, connectTo});
}

void Module::addInput(const Port &inputPort) {
  inputs.push_back(inputPort);
}

void Module::addOutput(const Port &outputPort) {
  outputs.push_back(outputPort);
}

void Module::printBody(std::ostream &out) const {
  out << body;
}

void Module::printEmptyLine(std::ostream &out) const {
  out << "\n";
}

/*void Module::addWire(const Wire &inputWire) {
  wires.push_back(inputWire);
}*/

void Module::addBody(const std::string &body) {
  this->body = body;
}

ExternalModule::ExternalModule(const model::NodeType* nodetype) : Module() {
  moduleName = nodetype->name;
  addInput(Port("clock", true, 1, "clock"));
  addInput(Port("reset", true, 1, "reset"));
  for (const auto *input : nodetype->inputs) {
    addInput(Port(input->name, true));
  }
  for (const auto *output : nodetype->outputs) {
    addOutput(Port(output->name, false));
  }
  auto meta = Library::get().find(*nodetype);
  auto element = meta->construct(meta->params);
  addBody(element->ir);
}

void ExternalModule::printDeclaration(std::ostream &out) const {
  out << "module " << moduleName << "(\n";
  bool hasComma = false;
  for (const auto &input : inputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << input.name;
    if (input.width != 1) {
      out << "[" << input.width << ":" << 0 << "],\n";
    }
    hasComma = true;
  }
  for (const auto &output : outputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << output.name;
    if (output.width != 1) {
      out << "[" << output.width << ":" << 0 << "],\n";
    }
    hasComma = true;
  }
  out << ");\n";
}

void ExternalModule::printEpilogue(std::ostream &out) const {
  out << "endmodule " << "//" << moduleName;
  printEmptyLine(out);
}

void ExternalModule::printVerilog(std::ostream &out) const {
  printDeclaration(out);
  printBody(out);
  printEpilogue(out);
  printEmptyLine(out);
}

void ExternalModule::printFirrtlDeclaration(std::ostream &out) const {
  out << Compiler::indent << Compiler::opPrefix << "extmodule @" <<
  moduleName << "(";
  bool hasComma = false;
  for (const auto &input : inputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << Compiler::indent <<  "in " <<
        input.name << " : " << Compiler::typePrefix << input.type;
    if (input.type != "reset" && input.type != "clock") {
      out << "<" << input.width << ">";
    }
    hasComma = true;
  }
  for (const auto &output : outputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << Compiler::indent <<  "out " <<  output.name <<
        ": " << Compiler::typePrefix << output.type << "<" << output.width <<
        ">";
  }
  out << ")\n";
}

FirrtlModule::FirrtlModule(const eda::hls::model::Model &model) : Module() {
  const auto* graph = model.main();
  moduleName = graph->name;
  //Inputs & Outputs
  addInput(Port("clock", true, 1, "clock"));
  addInput(Port("reset", true, 1, "reset"));
  for (const auto *node : graph->nodes) {
    if (node->isSource()) {
      for (const auto *output : node->outputs) {
        addInput(Port(output->name, true));
      }
    }
    if (node->isSink()) {
      for (const auto *input : node->inputs) {
        addOutput(Port(input->name, false));
      }
    }
  }
  //Instances
  for (const auto *node : graph->nodes) {
    addInstance(Instance(node->name,
                         node->type.name));
    for (const auto *input : node->inputs) {
      Port inputPort(node->name + "_" + input->target.port->name, true);
      instances.back().addInput(inputPort);
    }
    for (const auto *moduleInput : node->type.inputs) {
      Port inputPort(moduleInput->name, true);
      instances.back().addModuleInput(inputPort);
    }

    for (const auto *output : node->outputs) {
      Port outputPort(node->name + "_" + output->source.port->name, false);
      instances.back().addOutput(outputPort);
      if (node->type.name != "sink") {
        Port connectsTo(output->target.node->name + "_" +
            output->target.port->name, true);
        instances.back().addBinding(outputPort, connectsTo);
      }
    }
    for (const auto *moduleOutput : node->type.outputs) {
      Port outputPort(moduleOutput->name, false);
      instances.back().addModuleOutput(outputPort);
    }
    addBody("");
  }
}

void FirrtlModule::addInstance(const Instance &inputInstance) {
  instances.push_back(inputInstance);
}

/*void Module::printWires(std::ostream &out) const {
  for (const auto &wire : wires) {
    out << Compiler::indent << Compiler::indent << Compiler::varPrefix <<
        wire.name << " = " << Compiler::opPrefix << "wire :" <<
        Compiler::typePrefix << wire.type;
    if (wire.type != "clock" && wire.type != "reset") {
      out << "<" << wire.width << ">";
    }
    out << "\n";
  }
}*/

void FirrtlModule::printInstances(std::ostream &out) const {
  for (const auto &instance : instances) {
    out << Compiler::indent << Compiler::indent;
    out << Compiler::varPrefix << instance.instanceName << "_" << "clock" <<
        "," << " ";
    out << Compiler::varPrefix << instance.instanceName << "_" << "reset" <<
        "," << " ";
    bool hasComma = false;
    for (const auto &input : instance.inputs) {
      out << (hasComma ? ", " : "");
      out << Compiler::varPrefix << input.name;
      hasComma = true;
    }

    for (const auto &output : instance.outputs) {
      out << (hasComma ? ", " : "");
      out << Compiler::varPrefix << output.name;
      hasComma = true;
    }

    out << " = " << Compiler::opPrefix << "instance " << instance.instanceName
        << " @" << instance.moduleName << "(";
    hasComma = false;
    out << "in " << "clock " << ": " << Compiler::typePrefix << "clock, ";
    out << "in " << "reset " << ": " << Compiler::typePrefix << "reset,";
    for (const auto &input : instance.moduleInputs) {
      out << (hasComma ? ", " : " ");
      out << "in " << input.name << ": ";
      out << Compiler::typePrefix << input.type;
      out << "<" << input.width << ">";
      hasComma = true;
    }

    for (const auto &output : instance.moduleOutputs) {
      out << (hasComma ? ", " : " ");
      out << "out " << output.name << ": ";
      out << Compiler::typePrefix << output.type;
      out << "<" << output.width << ">";
      hasComma = true;
    }

    out << ")\n";
  }
}

void FirrtlModule::printConnections(std::ostream &out) const {
  for (const auto &instance : instances) {
    out << Compiler::indent << Compiler::indent << Compiler::opPrefix <<
        "connect " << Compiler::varPrefix << instance.instanceName + "_" +
        "clock, " << Compiler::varPrefix << "clock" << " : " <<
        Compiler::typePrefix << "clock" << ", " << Compiler::typePrefix <<
        "clock\n";
    out << Compiler::indent << Compiler::indent << Compiler::opPrefix <<
        "connect " << Compiler::varPrefix << instance.instanceName + "_" +
        "reset, " << Compiler::varPrefix << "reset" << " : " <<
        Compiler::typePrefix << "reset" << ", " << Compiler::typePrefix <<
        "reset\n";
    for (const auto &pair : instance.bindings) {
      out << Compiler::indent << Compiler::indent << Compiler::opPrefix <<
          "connect " << Compiler::varPrefix << pair.second.name << ", "
          << Compiler::varPrefix << pair.first.name;
      out <<  " : " << Compiler::typePrefix << pair.second.type;
      out << "<" << pair.second.width << ">";
      out << ", ";
      out << Compiler::typePrefix << pair.second.type;
      out << "<" << pair.second.width << ">";
      out << "\n";
    }
  }
}

void FirrtlModule::printDeclaration(std::ostream &out) const {
  out << Compiler::indent << Compiler::opPrefix << "module @" << moduleName <<
      " (\n";
  for (const auto &input : inputs) {
    out << Compiler::indent << Compiler::indent <<  "in " <<
        Compiler::varPrefix << input.name << " : " << Compiler::typePrefix <<
        input.type;
    if (input.name != "clock" && input.name != "reset") {
      out << "<" << input.width << ">,\n";
    } else {
      out << ",\n";
    }
  }
  bool hasComma = false;
  for (const auto &output : outputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << Compiler::indent <<  "out " <<
        Compiler::varPrefix << output.name << ": " << Compiler::typePrefix <<
        output.type << "<" << output.width << ">";
    hasComma = true;
  }
  out << ")\n";
  out << Compiler::indent << Compiler::indent << "{\n";
}

void FirrtlModule::printEpilogue(std::ostream &out) const {
  out << Compiler::indent << Compiler::indent << "} ";
  printEmptyLine(out);
}

void FirrtlModule::printFirrtl(std::ostream &out) const {
  printDeclaration(out);
  printEmptyLine(out);
  //printWires(out);
  printEmptyLine(out);
  printInstances(out);
  printConnections(out);
  printBody(out);
  printEpilogue(out);
  printEmptyLine(out);
}

Circuit::Circuit(std::string moduleName) : name(moduleName) {}

void Circuit::addFirModule(const FirrtlModule &firModule) {
  firModules.insert({firModule.moduleName, firModule});
}

void Circuit::addExternalModule(const ExternalModule &externalModule) {
  extModules.insert({externalModule.moduleName, externalModule});
}

FirrtlModule* Circuit::findFirModule(const std::string &name) const {
  auto i = std::find_if(firModules.begin(), firModules.end(),
      [&name](std::pair<std::string, FirrtlModule> const &pair) {
          return pair.first == name; });

  return const_cast<FirrtlModule*>
      (i != firModules.end() ? &(i->second) : nullptr);
}

ExternalModule* Circuit::findExtModule(const std::string &name) const {
  auto i = std::find_if(extModules.begin(), extModules.end(),
      [&name](std::pair<std::string, ExternalModule> const &pair) {
          return pair.first == name; });

  return const_cast<ExternalModule*>
      (i != extModules.end() ? &(i->second) : nullptr);
}

Module* Circuit::findModule(const std::string &name) const {

  Module *firModule = findFirModule(name);
  Module *extModule = findExtModule(name);

  return firModule == nullptr ?
      (extModule == nullptr ? nullptr : extModule) : firModule;
}

Module* Circuit::findMain() const {
  return findModule("main");
}

void Circuit::printFirrtl(std::ostream &out) const {
  out << Compiler::opPrefix << "circuit " << "\"" << name <<
      "\" " << "{\n";
  for (const auto &pair : firModules) {
    pair.second.printFirrtl(out);
  }
  for (const auto &pair : extModules) {
    pair.second.printFirrtlDeclaration(out);
  }
  out << "}";
}

void Circuit::printVerilog(std::ostream &out) const {
  for (const auto &pair : extModules) {
    pair.second.printVerilog(out);
  }
}

void Circuit::convertToSV(const std::string& inputFirrtlName) const {
  system((std::string(Compiler::circt) +
          inputFirrtlName +
          std::string(Compiler::circt_options)).c_str());
}

void Circuit::printFiles(const std::string& outputFirrtlName,
                         const std::string& outputVerilogName,
                         const std::string& outputDirectoryName) const {
  std::ofstream outputFile;
  outputFile.open(outputDirectoryName + outputFirrtlName);
  printFirrtl(outputFile);
  outputFile.close();
  convertToSV(outputDirectoryName + outputFirrtlName);
  std::filesystem::create_directory(outputDirectoryName);
  std::filesystem::rename("main.sv", (outputDirectoryName +
                                        std::string("main.sv")).c_str());
  outputFile.open(outputDirectoryName + outputVerilogName);
  printVerilog(outputFile);
  outputFile.close();
}

std::ostream& operator <<(std::ostream &out, const Circuit &circuit) {
  circuit.printFirrtl(out);
  circuit.printVerilog(out);
  return out;
}

std::shared_ptr<Circuit> Compiler::constructCircuit() {
  circuit = std::make_shared<Circuit>(std::string(model->main()->name));
  for (const auto *nodetype : model->nodetypes) {
    circuit->addExternalModule(nodetype);
  }
  circuit->addFirModule(*model);
  return circuit;
}

Compiler::Compiler(const Model &model) :
  model(std::make_shared<Model>(model)) {}

void Compiler::printRndVlogTest(const std::string& tstPath, const int tstCnt) {

  const std::filesystem::path path = tstPath;
  std::filesystem::create_directories(path.parent_path());

  std::ofstream tBenchFile(path);

  ctemplate::TemplateDictionary *dict = new ctemplate::TemplateDictionary("tb");

  // set generation time
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

  const Module *main = circuit->findMain();
  dict->SetValue("MODULE_NAME", main->moduleName);

  std::vector<std::string> bndNames;
  // set registers for device inputs
  std::vector<Port> inputs = main->inputs;
  for (size_t i = 0; i < inputs.size(); i++) {
    ctemplate::TemplateDictionary *inDict = dict->AddSectionDictionary("INS");
    inDict->SetValue("IN_TYPE", "[0:0]"); // TODO: set type when implemented
    const std::string iName = inputs[i].name;
    inDict->SetValue("IN_NAME", iName);
    bndNames.push_back(iName);
  }

  // set registers for device outputs
  std::vector<Port> outputs = main->outputs;
  for (size_t i = 0; i < outputs.size(); i++) {
    ctemplate::TemplateDictionary *outDict = dict->AddSectionDictionary("OUTS");
    outDict->SetValue("OUT_TYPE", "[0:0]"); // TODO: set type when implemented
    const std::string oName = outputs[i].name;
    outDict->SetValue("OUT_NAME", oName);
    bndNames.push_back(oName);
  }

  // calculate model's latency and store it
  DijkstraBalancer::get().balance(*model);
  const int latency = DijkstraBalancer::get().getGraphLatency();
  dict->SetIntValue("LATENCY", latency);

  // set bindings to device under test
  for (size_t i = 0; i < bndNames.size(); i++) {
    ctemplate::TemplateDictionary *bndDict = dict->AddSectionDictionary("BIND");
    bndDict->SetValue("WNAME", bndNames[i]);
    bndDict->SetValue("SEP", (i == bndNames.size() - 1) ? "" : ",\n");
  }
  bndNames.clear();

  // generate random stimuli
  for (size_t i = 0; i < (long unsigned)tstCnt; i++) {
    ctemplate ::TemplateDictionary *tDict = dict->AddSectionDictionary("TESTS");
    for (size_t j = 0; j < inputs.size(); j++) {

      if (inputs[j].isClock())
          continue;

      // TODO: set random values for inputs
      ctemplate ::TemplateDictionary *sDict = tDict->AddSectionDictionary("ST");
      sDict->SetValue("NAME", inputs[j].name);
    }
  }

  // use the template to store testbench to file
  std::string output;
  const std::string vlogTpl = "./src/data/ctemplate/tbench_verilog.tpl";
  ctemplate::ExpandTemplate(vlogTpl, ctemplate::DO_NOT_STRIP, dict, &output);
  tBenchFile << output;
  tBenchFile.close();
}

} // namespace eda::hls::compiler
