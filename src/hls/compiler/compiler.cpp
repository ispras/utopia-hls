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

/*void Module::addWire(const Wire &inputWire) {
  wires.push_back(inputWire);
}*/

void Module::addBody(const std::string &body) {
  this->body = body;
}

ExternalModule::ExternalModule(const model::NodeType* nodetype) : Module(nodetype->name) {
  addInput(Port("clock", true, Type("clock", 1, false, 1)));
  addInput(Port("reset", true, Type("reset", 1, false, 1)));
  for (const auto *input : nodetype->inputs) {
      /*const FixedType* fixed = (const FixedType*)&(input->type);*/
      addInput(Port(input->name, true, Type("sint",
                                            16,
                                            false,
                                            1)));
  }
  for (const auto *output : nodetype->outputs) {
      /*const FixedType* fixed = (const FixedType*)&(output->type);*/
      addOutput(Port(output->name, false, Type("sint",
                                               16,
                                               false,
                                               1)));
  }
  auto meta = Library::get().find(*nodetype);
  auto element = meta->construct(meta->params);
  addBody(element->ir);
}

FirrtlModule::FirrtlModule(const eda::hls::model::Model &model, const std::string &topModuleName) : Module(topModuleName) {
  const auto* graph = model.findGraph(topModuleName);
  //moduleName = graph->name;
  //Inputs & Outputs
  addInput(Port("clock", true, Type("clock", 1, false, 1)));
  addInput(Port("reset", true, Type("reset", 1, false, 1)));
  for (const auto *node : graph->nodes) {
    if (node->isSource()) {
      for (const auto *output : node->outputs) {
        /*const FixedType* fixed = (const FixedType*)&(output->type);*/
        Port outputPort (node->name + "_" + output->target.port->name, true,
                         Type("sint",
                              16,
                              false,
                              1));
        addInput(outputPort);
      }
    }
    if (node->isSink()) {
      for (const auto *input : node->inputs) {
        /*const FixedType* fixed = (const FixedType*)&(input->type);*/
        Port inputPort(node->name + "_" + input->target.port->name, false,
                        Type("sint",
                             16,
                             false,
                             1));
        addOutput(Port(inputPort));
      }
    }
  }
  //Instances
  for (const auto *node : graph->nodes) {
    //Skip dummies
    if (node->isSource() || node->isSink()) {
      continue;
    }
    addInstance(Instance(node->name,
                         node->type.name));
    for (const auto *input : node->inputs) {
      /*const FixedType* fixed = (const FixedType*)&(input->type);*/
      Port inputPort(node->name + "_" + input->target.port->name, true,
                       Type("sint",
                            16,
                            false,
                            1));
      instances.back().addInput(inputPort);
      if (input->source.node->isSource()) {
        Port connectsFrom(input->source.node->name + "_" +
            input->source.port->name, true, Type("sint",
                                                  16,
                                                  false,
                                                  1));
        instances.back().addBinding(connectsFrom, inputPort);
      }
    }
    for (const auto *moduleInput : node->type.inputs) {
      /*const FixedType* fixed = (const FixedType*)&(moduleInput->type);*/
      Port inputPort(moduleInput->name, true, Type("sint",
                                                   16,
                                                   false,
                                                   1));
      instances.back().addModuleInput(inputPort);
    }

    for (const auto *output : node->outputs) {
      /*const FixedType* fixed = (const FixedType*)&(output->type);*/
      Port outputPort(node->name + "_" + output->source.port->name, false,
                      Type("sint",
                           16,
                           false,
                           1));
      instances.back().addOutput(outputPort);
      if (node->type.name != "sink") {
        Port connectsTo(output->target.node->name + "_" +
            output->target.port->name, true, Type("sint",
                                                  16,
                                                  false,
                                                  1));
        instances.back().addBinding(outputPort, connectsTo);
      }
    }
    for (const auto *moduleOutput : node->type.outputs) {
      /*const FixedType* fixed = (const FixedType*)&(moduleOutput->type);*/
      Port outputPort(moduleOutput->name, false, Type("sint",
                                                      16,
                                                      false,
                                                      1));
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

Circuit::Circuit(const std::string &moduleName) : name(moduleName) {}

void Circuit::addFirModule(const FirrtlModule &firModule) {
  firModules.insert({firModule.moduleName, firModule});
}

void Circuit::addExternalModule(const ExternalModule &externalModule) {
  extModules.insert({externalModule.moduleName, externalModule});
}

FirrtlModule* Circuit::findFirModule(const std::string &name) const {
  auto iter = firModules.find(name);

  return const_cast<FirrtlModule*>(
      (iter == firModules.end()) ? nullptr : &(iter->second));
}

ExternalModule* Circuit::findExtModule(const std::string &name) const {
  auto iter = extModules.find(name);

  return const_cast<ExternalModule*>(
      (iter == extModules.end()) ? nullptr : &(iter->second));
}

Module* Circuit::findModule(const std::string &name) const {

  Module *firModule = findFirModule(name);
  Module *extModule = findExtModule(name);

  return firModule == nullptr ?
      (extModule == nullptr ? nullptr : extModule) : firModule;
}

Module* Circuit::findMain() const {
  return findModule(name);
}

void Compiler::printInstances(const FirrtlModule &firmodule,
                              std::ostream &out) const {
  for (const auto &instance : firmodule.instances) {
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
      out << Compiler::typePrefix << input.type.name;
      out << "<" << input.type.element_width << ">";
      hasComma = true;
    }

    for (const auto &output : instance.moduleOutputs) {
      out << (hasComma ? ", " : " ");
      out << "out " << output.name << ": ";
      out << Compiler::typePrefix << output.type.name;
      out << "<" << output.type.element_width << ">";
      hasComma = true;
    }

    out << ")\n";
  }
}

void Compiler::printConnections(const FirrtlModule &firmodule,
                                std::ostream &out) const {
  for (const auto &instance : firmodule.instances) {
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
      out <<  " : " << Compiler::typePrefix << pair.second.type.name;
      out << "<" << pair.second.type.element_width << ">";
      out << ", ";
      out << Compiler::typePrefix << pair.second.type.name;
      out << "<" << pair.second.type.element_width << ">";
      out << "\n";
    }
  }
}

void Compiler::printDeclaration(const FirrtlModule &firmodule,
                                std::ostream &out) const {
  out << Compiler::indent << Compiler::opPrefix << "module @" <<
      firmodule.moduleName << " (\n";
  for (const auto &input : firmodule.inputs) {
    out << Compiler::indent << Compiler::indent <<  "in " <<
        Compiler::varPrefix << input.name << " : " << Compiler::typePrefix <<
        input.type.name;
    if (input.name != "clock" && input.name != "reset") {
      out << "<" << input.type.element_width << ">,\n";
    } else {
      out << ",\n";
    }
  }
  bool hasComma = false;
  for (const auto &output : firmodule.outputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << Compiler::indent <<  "out " <<
        Compiler::varPrefix << output.name << ": " << Compiler::typePrefix <<
        output.type.name << "<" << output.type.element_width << ">";
    hasComma = true;
  }
  out << ")\n";
  out << Compiler::indent << Compiler::indent << "{\n";
}

void Compiler::printEpilogue(const FirrtlModule &firmodule,
                             std::ostream &out) const {
  out << Compiler::indent << Compiler::indent << "} ";
  printEmptyLine(out);
}

void Compiler::printFirrtlModule(const FirrtlModule &firmodule,
                                 std::ostream &out) const {
  printDeclaration(firmodule, out);
  printEmptyLine(out);
  //printWires(firmodule, out);
  printEmptyLine(out);
  printInstances(firmodule, out);
  printConnections(firmodule, out);
  printBody(firmodule, out);
  printEpilogue(firmodule, out);
  printEmptyLine(out);
}

void Compiler::printDeclaration(const ExternalModule &extmodule,
                                std::ostream &out) const {
  out << "module " << extmodule.moduleName << "(\n";
  bool hasComma = false;
  for (const auto &input : extmodule.inputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << input.name;
    /*if (input.type.element_width != 1) {
      out << "[" << input.type.element_width - 1 << ":" << 0 << "]";
    }*/
    hasComma = true;
  }
  for (const auto &output : extmodule.outputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << output.name;
    /*if (output.type.element_width != 1) {
      out << "[" << output.type.element_width - 1 << ":" << 0 << "]";
    }*/
    hasComma = true;
  }
  out << ");\n";
}

void Compiler::printEpilogue(const ExternalModule &extmodule,
                             std::ostream &out) const {
  out << "endmodule " << "//" << extmodule.moduleName;
  printEmptyLine(out);
}

void Compiler::printVerilogModule(const ExternalModule &extmodule,
                                  std::ostream &out) const {
  printDeclaration(extmodule, out);
  printBody(extmodule, out);
  printEpilogue(extmodule, out);
  printEmptyLine(out);
}

void Compiler::printFirrtlDeclaration(const ExternalModule &extmodule,
                                      std::ostream &out) const {
  out << Compiler::indent << Compiler::opPrefix << "extmodule @" <<
  extmodule.moduleName << "(";
  bool hasComma = false;
  for (const auto &input : extmodule.inputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << Compiler::indent <<  "in " <<
        input.name << " : " << Compiler::typePrefix << input.type.name;
    if (input.type.name != "reset" && input.type.name != "clock") {
      out << "<" << input.type.element_width << ">";
    }
    hasComma = true;
  }
  for (const auto &output : extmodule.outputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Compiler::indent << Compiler::indent <<  "out " <<  output.name <<
        ": " << Compiler::typePrefix << output.type.name << "<" <<
         output.type.element_width << ">";
  }
  out << ")\n";
}

void Compiler::printBody(const Module &module, std::ostream &out) const {
  out << module.body;
}

void Compiler::printEmptyLine(std::ostream &out) const {
  out << "\n";
}

void Compiler::printFirrtl(std::ostream &out) const {
  out << Compiler::opPrefix << "circuit " << "\"" << circuit->name <<
      "\" " << "{\n";
  for (const auto &pair : circuit->firModules) {
    printFirrtlModule(pair.second, out);
  }
  for (const auto &pair : circuit->extModules) {
    printFirrtlDeclaration(pair.second, out);
  }
  out << "}";
}

void Compiler::printVerilog(std::ostream &out) const {
  for (const auto &pair : circuit->extModules) {
    printVerilogModule(pair.second, out);
  }
}

void Compiler::convertToSV(const std::string& inputFirrtlName) const {
  system((std::string(Compiler::circt) +
          inputFirrtlName +
          std::string(Compiler::circt_options)).c_str());
}

void Compiler::printFiles(const std::string& outputFirrtlName,
                          const std::string& outputVerilogName,
                          const std::string& outputDirectoryName) const {
  std::filesystem::create_directory(outputDirectoryName);
  std::ofstream outputFile;
  outputFile.open(outputDirectoryName + outputFirrtlName);
  printFirrtl(outputFile);
  outputFile.close();
  convertToSV(outputDirectoryName + outputFirrtlName);
  std::filesystem::rename("main.sv", (outputDirectoryName +
                                      std::string("main.sv")).c_str());
  outputFile.open(outputDirectoryName + outputVerilogName);
  printVerilog(outputFile);
  outputFile.close();
}

std::shared_ptr<Circuit> Compiler::constructCircuit(const std::string& topModuleName) {
  circuit = std::make_shared<Circuit>(topModuleName);

  for (const auto *nodetype : model->nodetypes) {
    circuit->addExternalModule(nodetype);
  }
  circuit->addFirModule(FirrtlModule(*model, topModuleName));

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
    if (inputs[i].name == "clock" || inputs[i].name == "reset") {
      inDict->SetValue("IN_TYPE", ""); // TODO: set type when implemented
    } else {
      inDict->SetValue("IN_TYPE", "[15:0]"); // TODO: set type when implemented
    }
    const std::string iName = inputs[i].name;
    inDict->SetValue("IN_NAME", iName);
    bndNames.push_back(iName);
  }

  // set registers for device outputs
  std::vector<Port> outputs = main->outputs;
  for (size_t i = 0; i < outputs.size(); i++) {
    ctemplate::TemplateDictionary *outDict = dict->AddSectionDictionary("OUTS");
    outDict->SetValue("OUT_TYPE", "[15:0]"); // TODO: set type when implemented
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
