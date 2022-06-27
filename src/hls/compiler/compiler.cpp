//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/compiler/compiler.h"
#include "hls/scheduler/dijkstra.h"
#include "util/assert.h"

#include <filesystem>
#include <fstream>

using namespace eda::hls::mapper;
using namespace eda::hls::model;
using namespace eda::hls::scheduler;
using namespace eda::hls::library;
using namespace eda::utils;

namespace eda::hls::compiler {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

// TODO: Think about this
void replaceSomeChars(std::string &buf) {
  std::replace(buf.begin(), buf.end(), ',', '_');
  std::replace(buf.begin(), buf.end(), '>', '_');
  std::replace(buf.begin(), buf.end(), '<', '_');
}

const std::string chanSourceToString(const eda::hls::model::Chan &chan) {
  return chan.source.node->name + "_" + chan.source.port->name;
}

Port createFirrtlPort(const model::Node *node,
                      const model::Port *port,
                      const Port::Direction dir,
                      const Type type) {
  return Port(node->name + "_" + port->name, dir, type);
}

bool Port::isClock() const {
  return name == "clock";
}

std::string ExternalModule::getFileNameFromPath(const std::string &path) {
  int start = 0;
  int end = path.find("/");
  std::string buf = "";
  std::string outputFileName;
  while (end != -1) {
    buf = path.substr(start, end - start);
    start = end + 1;
    end = path.find("/", start);
  }
  buf = path.substr(start, end - start);
  return buf;
}

void ExternalModule::addInputs(const std::vector<model::Port*> inputs) {
  //Will be removed in the near future.
  addInput(Port("clock", Port::Direction::IN, Type("clock", 1)));
  addInput(Port("reset", Port::Direction::IN, Type("reset", 1)));
  for (const auto *input : inputs) {
      addInput(Port(input->name, Port::Direction::IN, Type("sint", 16)));
  }
}

void ExternalModule::addOutputs(const std::vector<model::Port*> outputs) {
  for (const auto *output : outputs) {
      addOutput(Port(output->name, Port::Direction::OUT, Type("sint", 16)));
  }
}

void FirrtlModule::addInputs(const model::Node *node,
                             const std::vector<model::Chan*> outputs) {
  for (const auto *output : outputs) {
    addInput(createFirrtlPort(node, output->source.port, Port::Direction::IN,
      Type("sint", 16)));
  }
}

void FirrtlModule::addOutputs(const model::Node *node,
                              const std::vector<model::Chan*> inputs) {
  for (const auto *input : inputs) {
    addOutput(createFirrtlPort(node, input->target.port, Port::Direction::OUT,
      Type("sint", 16)));
  }
}

void Instance::addModuleInputs(const std::vector<model::Port*> inputs) {
  for (const auto *input : inputs) {
    addModuleInput(Port(input->name, Port::Direction::IN,
      Type("sint", 16)));
  }
}

void Instance::addModuleOutputs(const std::vector<model::Port*> outputs) {
  for (const auto *output : outputs) {
    addModuleOutput(Port(output->name, Port::Direction::OUT,
      Type("sint", 16)));
  }
}

ExternalModule::ExternalModule(const model::NodeType *nodetype) :
    Module(nodetype->name) {
  addInputs(nodetype->inputs);
  addOutputs(nodetype->outputs);

  auto metaElement = Library::get().find(*nodetype, HWConfig("", "", ""));
  // TODO: what if the element in not found?

  auto element = metaElement->construct(metaElement->params);
  setBody(element->ir);
  setPath(element->path);
}

FirrtlModule::FirrtlModule(const eda::hls::model::Model &model,
  const std::string &topModuleName) : Module(topModuleName) {
  const auto* graph = model.findGraph(topModuleName);

  //Inputs & Outputs
  addInput(Port("clock", Port::Direction::IN, Type("clock", 1)));
  addInput(Port("reset", Port::Direction::IN, Type("reset", 1)));

  for (const auto *node : graph->nodes) {
    if (node->isSource()) {
      addInputs(node, node->outputs);
    } else if (node->isSink()) {
      addOutputs(node, node->inputs);
    } else {
      Instance instance(node->name, node->type.name);

      for (const auto *input : node->inputs) {
        Port inputPort = createFirrtlPort(node,
          input->target.port, Port::Direction::IN, Type("sint", 16));
        instance.addInput(inputPort);

        if (input->source.node->isSource()) {
          Port connectsFrom = createFirrtlPort(input->source.node,
            input->source.port, Port::Direction::OUT, Type("sint", 16));
          instance.addBinding(connectsFrom, inputPort);
        }
      }

      instance.addModuleInputs(node->type.inputs);

      for (const auto *output : node->outputs) {
        Port outputPort = createFirrtlPort(node,
          output->source.port, Port::Direction::OUT, Type("sint", 16));
        instance.addOutput(outputPort);

        if (!node->type.isSink()) { // check this
          Port connectsTo = createFirrtlPort(output->target.node,
            output->target.port, Port::Direction::IN, Type("sint", 16));
          instance.addBinding(outputPort, connectsTo);
        }
      }

      instance.addModuleOutputs(node->type.outputs);

      addInstance(instance);
      setPath(""); // ???
    }
  }
}

void FirrtlModule::printInstances(std::ostream &out) const {
  for (const auto &instance : instances) {
    out << FirrtlCircuit::indent << FirrtlCircuit::indent;
    out << FirrtlCircuit::varPrefix << instance.instanceName << "_clock, ";
    out << FirrtlCircuit::varPrefix << instance.instanceName << "_reset, ";
    bool hasComma = false;
    std::string buf;
    for (const auto &input : instance.inputs) {
      out << (hasComma ? ", " : "");
      buf.assign(input.name);
      replaceSomeChars(buf);
      out << FirrtlCircuit::varPrefix << buf;
      hasComma = true;
    }

    for (const auto &output : instance.outputs) {
      out << (hasComma ? ", " : "");
      buf.assign(output.name);
      replaceSomeChars(buf);
      out << FirrtlCircuit::varPrefix << buf;
      hasComma = true;
    }

    out << " = " << FirrtlCircuit::opPrefix << "instance " <<
        instance.instanceName   << " @" << instance.moduleName << "(";
    hasComma = false;
    out << "in " << "clock " << ": " << FirrtlCircuit::typePrefix << "clock, ";
    out << "in " << "reset " << ": " << FirrtlCircuit::typePrefix << "reset,";
    for (const auto &input : instance.moduleInputs) {
      out << (hasComma ? ", " : " ");
      buf.assign(input.name);
      replaceSomeChars(buf);
      out << "in " << buf << ": ";
      out << FirrtlCircuit::typePrefix << input.type.name;
      out << "<" << input.type.width << ">";
      hasComma = true;
    }

    for (const auto &output : instance.moduleOutputs) {
      out << (hasComma ? ", " : " ");
      buf.assign(output.name);
      replaceSomeChars(buf);
      out << "out " << buf << ": ";
      out << FirrtlCircuit::typePrefix << output.type.name;
      out << "<" << output.type.width << ">";
      hasComma = true;
    }

    out << ")\n";
  }
}

void FirrtlModule::printConnections(std::ostream &out) const {
  for (const auto &instance : instances) {
    out << FirrtlCircuit::indent << FirrtlCircuit::indent <<
        FirrtlCircuit::opPrefix << "connect " << FirrtlCircuit::varPrefix <<
        instance.instanceName + "_" + "clock, " << FirrtlCircuit::varPrefix <<
        "clock" << " : " << FirrtlCircuit::typePrefix << "clock" << ", " <<
        FirrtlCircuit::typePrefix << "clock\n";
    out << FirrtlCircuit::indent << FirrtlCircuit::indent <<
         FirrtlCircuit::opPrefix << "connect " << FirrtlCircuit::varPrefix <<
         instance.instanceName + "_" + "reset, " << FirrtlCircuit::varPrefix <<
         "reset" << " : " << FirrtlCircuit::typePrefix << "reset" << ", " <<
         FirrtlCircuit::typePrefix << "reset\n";
    std::string buf1;
    std::string buf2;
    for (const auto &pair : instance.bindings) {
      buf1.assign(pair.first.name);
      replaceSomeChars(buf1);
      buf2.assign(pair.second.name);
      replaceSomeChars(buf2);
      out << FirrtlCircuit::indent << FirrtlCircuit::indent <<
          FirrtlCircuit::opPrefix << "connect " << FirrtlCircuit::varPrefix <<
          buf2 << ", " << FirrtlCircuit::varPrefix << buf1;
      out <<  " : " << FirrtlCircuit::typePrefix << pair.second.type.name;
      out << "<" << pair.second.type.width << ">";
      out << ", ";
      out << FirrtlCircuit::typePrefix << pair.second.type.name;
      out << "<" << pair.second.type.width << ">";
      out << "\n";
    }
  }
}

void FirrtlModule::printDeclaration(std::ostream &out) const {
  out << FirrtlCircuit::indent << FirrtlCircuit::opPrefix << "module @" <<
      name << " (\n";
  for (const auto &input : inputs) {
    out << FirrtlCircuit::indent << FirrtlCircuit::indent <<  "in " <<
        FirrtlCircuit::varPrefix << input.name << " : " <<
        FirrtlCircuit::typePrefix << input.type.name;
    if (input.type.width != 1) {
      out << "<" << input.type.width << ">,\n";
    } else {
      out << ",\n";
    }
  }
  bool hasComma = false;
  for (const auto &output : outputs) {
    out << (hasComma ? ",\n" : "\n");
    out << FirrtlCircuit::indent << FirrtlCircuit::indent <<  "out " <<
        FirrtlCircuit::varPrefix << output.name << ": " <<
        FirrtlCircuit::typePrefix << output.type.name;
    if (output.type.width != 1) {
      out << "<" << output.type.width << ">";
    }
    hasComma = true;
  }
  out << ")\n";
  out << FirrtlCircuit::indent << FirrtlCircuit::indent << "{\n";
}

void FirrtlModule::printEpilogue(std::ostream &out) const {
  out << FirrtlCircuit::indent << FirrtlCircuit::indent << "} ";
}

void FirrtlModule::printFirrtlModule(std::ostream &out) const {
  printDeclaration(out);
  printInstances(out);
  printConnections(out);
  printEpilogue(out);
}

void ExternalModule::printDeclaration(std::ostream &out) const {
  out << "module " << name << "(\n";
  bool hasComma = false;
  std::string buf;
  for (const auto &input : inputs) {
    out << (hasComma ? ",\n" : "\n");
    buf.assign(input.name);
    replaceSomeChars(buf);
    out << FirrtlCircuit::indent << buf;
    /*if (input.type.element_width != 1) {
      out << "[" << input.type.element_width - 1 << ":" << 0 << "]";
    }*/
    hasComma = true;
  }
  for (const auto &output : outputs) {
    out << (hasComma ? ",\n" : "\n");
    buf.assign(output.name);
    replaceSomeChars(buf);
    out << FirrtlCircuit::indent << buf;
    /*if (output.type.element_width != 1) {
      out << "[" << output.type.element_width - 1 << ":" << 0 << "]";
    }*/
    hasComma = true;
  }
  out << ");\n";
}

void ExternalModule::printEpilogue(std::ostream &out) const {
  out << "endmodule " << "//" << name;
}

void ExternalModule::printFirrtlDeclaration(std::ostream &out) const {
  out << FirrtlCircuit::indent << FirrtlCircuit::opPrefix << "extmodule @" <<
  name << "(";
  bool hasComma = false;
  std::string buf;
  for (const auto &input : inputs) {
    buf.assign(input.name);
    replaceSomeChars(buf);
    out << (hasComma ? ",\n" : "\n");
    out << FirrtlCircuit::indent << FirrtlCircuit::indent <<  "in " <<
        buf << " : " << FirrtlCircuit::typePrefix << input.type.name;
    if (input.type.width != 1) {
      out << "<" << input.type.width << ">";
    }
    hasComma = true;
  }
  for (const auto &output : outputs) {
    buf.assign(output.name);
    replaceSomeChars(buf);
    out << (hasComma ? ",\n" : "\n");
    out << FirrtlCircuit::indent << FirrtlCircuit::indent <<  "out " <<  buf <<
        ": " << FirrtlCircuit::typePrefix << output.type.name;
    if (output.type.width != 1) {
      out << "<" << output.type.width << ">";
    }
  }
  out << ")\n";
}

void ExternalModule::moveVerilogModule(
    const std::string &outputDirName) const {
  std::string outputFileName = outputDirName + getFileNameFromPath(path);
  std::filesystem::copy(toLower(path),
                        outputFileName,
                        std::filesystem::copy_options::overwrite_existing);
}

void ExternalModule::printVerilogModule(std::ostream &out) const {
  printDeclaration(out);
  printBody(out);
  printEpilogue(out);
}

void FirrtlCircuit::printFirrtlModule(const FirrtlModule &firmodule,
                                      std::ostream &out) const {
  firmodule.printFirrtlModule(out);
}

void FirrtlCircuit::printDeclaration(std::ostream &out) const {
  out << FirrtlCircuit::opPrefix << "circuit " << "\"" << name <<
      "\" " << "{\n";
}

void FirrtlCircuit::printEpilogue(std::ostream &out) const {
  out << "}";
}

void FirrtlCircuit::printFirrtlModules(std::ostream &out) const {
  for (const auto &pair : firModules) {
    pair.second.printFirrtlModule(out);
  }
}

void FirrtlCircuit::printExtDeclarations(std::ostream &out) const {
  for (const auto &pair : extModules) {
    pair.second.printFirrtlDeclaration(out);
  }
}

void FirrtlCircuit::printFirrtl(std::ostream &out) const {
  printDeclaration(out);
  printFirrtlModules(out);
  printExtDeclarations(out);
  printEpilogue(out);
}

void FirrtlCircuit::moveVerilogLibrary(const std::string &outputDirName,
                                       std::ostream &out) const {
  for (const auto &pair : extModules) {
    if (pair.second.body == "") {
      pair.second.moveVerilogModule(outputDirName);
    } else {
      pair.second.printVerilogModule(out);
    }
  }
}

void FirrtlCircuit::convertToSV(const std::string& inputFirrtlName) const {
  assert((system((std::string(FirrtlCircuit::circt) +
         inputFirrtlName +
         std::string(FirrtlCircuit::circtOptions)).c_str()) != -1) &&
         "Error while creating top verilog module!");
}

void FirrtlCircuit::printFiles(const std::string& outputFirrtlName,
                               const std::string& outputVerilogLibraryName,
                               const std::string& outputVerilogTopModuleName,
                               const std::string& outputDirName) const {
  const std::filesystem::path outputPath = outputDirName;
  std::filesystem::create_directories(outputPath.parent_path());
  std::ofstream outputFile;
  const std::string outputPathStr = outputPath.string();
  outputFile.open(outputPathStr + outputFirrtlName);
  printFirrtl(outputFile);
  outputFile.close();
  convertToSV(outputPathStr + outputFirrtlName);
  std::filesystem::rename("main.sv", (outputPathStr +
                                     std::string(outputVerilogTopModuleName)).c_str());
  outputFile.open(outputPathStr + outputVerilogLibraryName);
  moveVerilogLibrary(outputPathStr, outputFile);
  outputFile.close();
}

std::shared_ptr<FirrtlCircuit> Compiler::constructCircuit(const Model &model,
    const std::string& name) {
  auto circuit = std::make_shared<FirrtlCircuit>(name);

  for (const auto *nodetype : model.nodetypes) {
    circuit->addExternalModule(nodetype);
  }
  circuit->addFirModule(FirrtlModule(model, name));

  return circuit;
}

void FirrtlCircuit::printRndVlogTest(const Model &model,
                                     const std::string &outputDirName,
                                     const std::string &outputTestName,
                                     const int latency,
                                     const size_t tstCnt) {
  const std::filesystem::path path = outputDirName + outputTestName;
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

  const auto main = (firModules.find("main"))->second;

  dict->SetValue("MODULE_NAME", main.name);

  std::vector<std::string> bndNames;
  // set registers for device inputs
  std::vector<Port> inputs = main.inputs;
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
  std::vector<Port> outputs = main.outputs;
  for (size_t i = 0; i < outputs.size(); i++) {
    ctemplate::TemplateDictionary *outDict = dict->AddSectionDictionary("OUTS");
    outDict->SetValue("OUT_TYPE", "[15:0]"); // TODO: set type when implemented
    const std::string oName = outputs[i].name;
    outDict->SetValue("OUT_NAME", oName);
    bndNames.push_back(oName);
  }

  // calculate model's latency and store it
  dict->SetIntValue("LATENCY", latency);

  // set bindings to device under test
  for (size_t i = 0; i < bndNames.size(); i++) {
    ctemplate::TemplateDictionary *bndDict = dict->AddSectionDictionary("BIND");
    bndDict->SetValue("WNAME", bndNames[i]);
    bndDict->SetValue("SEP", (i == bndNames.size() - 1) ? "" : ",\n");
  }
  bndNames.clear();

  // generate random stimuli
  for (size_t i = 0; i < tstCnt; i++) {
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
