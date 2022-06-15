//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
#include <filesystem>
#include <fstream>

#include "hls/compiler/compiler.h"

using namespace eda::hls::model;
using namespace eda::hls::scheduler;
using namespace eda::hls::library;

namespace eda::hls::compiler {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); }
                 );
    return s;
}

//TODO: Kill it with fire.
void hackFirrtl(std::string &buf) {
  std::replace( buf.begin(), buf.end(), ',', '_');
  std::replace( buf.begin(), buf.end(), '>', '_');
  std::replace( buf.begin(), buf.end(), '<', '_');
}

const std::string chanSourceToString(const eda::hls::model::Chan &chan) {
  return chan.source.node->name + "_" + chan.source.port->name;
}

Port createFirrtlPort(const model::Node *node,
                      const model::Port *port,
                      const Port::Direction dir,
                      const Type type) {
  Port firrtlPort(node->name + "_" + port->name, dir, type);
  return firrtlPort;
}

bool Port::isClock() const {
  return name == "clock";
}

void createDirRecur(const std::string& dirName) {
  int start = 0;
  int end = dirName.find("/");
  std::string dir = "";
  while (end != -1) {
    dir = dir + dirName.substr(start, end - start) + "/";
    if (!std::filesystem::exists(dir)) {
      assert(std::filesystem::create_directory(dir) &&
             "Error while creating output directory!");
    }
      start = end + 1;
      end = dirName.find("/", start);
  }
}

std::string getFileNameFromPath(const std::string &path) {
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
      /*const FixedType* fixed = (const FixedType*)&(input->type);*/
      addInput(Port(input->name, Port::Direction::IN, Type("sint", 16)));
  }
}

void ExternalModule::addOutputs(const std::vector<model::Port*> outputs) {
  for (const auto *output : outputs) {
      /*const FixedType* fixed = (const FixedType*)&(input->type);*/
      addOutput(Port(output->name, Port::Direction::OUT, Type("sint", 16)));
  }
}

void FirrtlModule::addInputs(const model::Node *node,
                             const std::vector<model::Chan*> outputs) {
  for (const auto *output : outputs) {
    /*const FixedType* fixed = (const FixedType*)&(output->type);*/
    Port outputPort = createFirrtlPort(node,
                                       output->source.port,
                                       Port::Direction::IN,
                                       Type("sint", 16));
    addInput(outputPort);
  }
}

void FirrtlModule::addOutputs(const model::Node *node,
                              const std::vector<model::Chan*> inputs) {
  /*const FixedType* fixed = (const FixedType*)&(output->type);*/
  for (const auto *input : inputs) {
    Port inputPort = createFirrtlPort(node,
                                      input->target.port,
                                      Port::Direction::OUT,
                                      Type("sint", 16));
    addOutput(inputPort);
  }
}

void Instance::addModuleInputs(const std::vector<model::Port*> inputs) {
  for (const auto *moduleInput : inputs) {
    /*const FixedType* fixed = (const FixedType*)&(moduleInput->type);*/
    Port inputPort(moduleInput->name,
                   Port::Direction::IN,
                   Type("sint", 16));
    addModuleInput(inputPort);
  }
}

void Instance::addModuleOutputs(const std::vector<model::Port*> outputs) {
  for (const auto *moduleOutput : outputs) {
    /*const FixedType* fixed = (const FixedType*)&(moduleInput->type);*/
    Port outputPort(moduleOutput->name,
                    Port::Direction::OUT,
                    Type("sint", 16));
    addModuleOutput(outputPort);
  }
}

ExternalModule::ExternalModule(const model::NodeType *nodetype) :
    Module(nodetype->name) {
  addInputs(nodetype->inputs);
  addOutputs(nodetype->outputs);
  auto metaElement = Library::get().find(*nodetype);
  auto element = metaElement->construct(metaElement->params);
  addBody(element->ir);
  addPath(element->path);
}

FirrtlModule::FirrtlModule(const eda::hls::model::Model &model,
                           const std::string &topModuleName) :
    Module(topModuleName) {
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
      /*addInstance(Instance(node->name,
                           node->type.name));*/
      for (const auto *input : node->inputs) {
        /*const FixedType* fixed = (const FixedType*)&(input->type);*/
        Port inputPort = createFirrtlPort(node,
                                          input->target.port,
                                          Port::Direction::IN,
                                          Type("sint", 16));
        instance.addInput(inputPort);
        if (input->source.node->isSource()) {
          Port connectsFrom = createFirrtlPort(input->source.node,
                                               input->source.port,
                                               Port::Direction::OUT,
                                               Type("sint", 16));
          instance.addBinding(connectsFrom, inputPort);
        }
      }
      instance.addModuleInputs(node->type.inputs);
      /*for (const auto *moduleInput : node->type.inputs) {
        Port inputPort(moduleInput->name, true, Type("sint",
                                                     16,
                                                     false,
                                                     1));
        instances.back().addModuleInput(inputPort);
      }*/

      for (const auto *output : node->outputs) {
        /*const FixedType* fixed = (const FixedType*)&(output->type);*/
        Port outputPort = createFirrtlPort(node,
                                           output->source.port,
                                           Port::Direction::OUT,
                                           Type("sint", 16));
        instance.addOutput(outputPort);
        if (!node->type.isSink()) {
          Port connectsTo = createFirrtlPort(output->target.node,
                                             output->target.port,
                                             Port::Direction::IN,
                                             Type("sint", 16));
          instance.addBinding(outputPort, connectsTo);
        }
      }
      instance.addModuleOutputs(node->type.outputs);
      /*for (const auto *moduleOutput : node->type.outputs) {
        Port outputPort(moduleOutput->name, false, Type("sint",
                                                        16,
                                                        false,
                                                        1));
        instances.back().addModuleOutput(outputPort);
      }*/
      addInstance(instance);
      addPath("");
    }
  }
}

void FirrtlModule::printInstances(std::ostream &out) const {
  for (const auto &instance : instances) {
    out << Circuit::indent << Circuit::indent;
    out << Circuit::varPrefix << instance.instanceName << "_" << "clock" <<
        "," << " ";
    out << Circuit::varPrefix << instance.instanceName << "_" << "reset" <<
        "," << " ";
    bool hasComma = false;
	  std::string buf;
    for (const auto &input : instance.inputs) {
      out << (hasComma ? ", " : "");
	    buf.assign(input.name);
      hackFirrtl(buf);
      out << Circuit::varPrefix << buf;
      hasComma = true;
    }

    for (const auto &output : instance.outputs) {
      out << (hasComma ? ", " : "");
	    buf.assign(output.name);
      hackFirrtl(buf);
      out << Circuit::varPrefix << buf;
      hasComma = true;
    }

    out << " = " << Circuit::opPrefix << "instance " << instance.instanceName
        << " @" << instance.moduleName << "(";
    hasComma = false;
    out << "in " << "clock " << ": " << Circuit::typePrefix << "clock, ";
    out << "in " << "reset " << ": " << Circuit::typePrefix << "reset,";
    for (const auto &input : instance.moduleInputs) {
      out << (hasComma ? ", " : " ");
	  buf.assign(input.name);
	  hackFirrtl(buf);
      out << "in " << buf << ": ";
      out << Circuit::typePrefix << input.type.name;
      out << "<" << input.type.element_width << ">";
      hasComma = true;
    }

    for (const auto &output : instance.moduleOutputs) {
      out << (hasComma ? ", " : " ");
	  buf.assign(output.name);
	  hackFirrtl(buf);
      out << "out " << buf << ": ";
      out << Circuit::typePrefix << output.type.name;
      out << "<" << output.type.element_width << ">";
      hasComma = true;
    }

    out << ")\n";
  }
}

void FirrtlModule::printConnections(std::ostream &out) const {
  for (const auto &instance : instances) {
    out << Circuit::indent << Circuit::indent << Circuit::opPrefix <<
        "connect " << Circuit::varPrefix << instance.instanceName + "_" +
        "clock, " << Circuit::varPrefix << "clock" << " : " <<
        Circuit::typePrefix << "clock" << ", " << Circuit::typePrefix <<
        "clock\n";
    out << Circuit::indent << Circuit::indent << Circuit::opPrefix <<
        "connect " << Circuit::varPrefix << instance.instanceName + "_" +
        "reset, " << Circuit::varPrefix << "reset" << " : " <<
        Circuit::typePrefix << "reset" << ", " << Circuit::typePrefix <<
        "reset\n";
	std::string buf1;
	std::string buf2;
  for (const auto &pair : instance.bindings) {
	  buf1.assign(pair.first.name);
	  hackFirrtl(buf1);
	  buf2.assign(pair.second.name);
	  hackFirrtl(buf2);
      out << Circuit::indent << Circuit::indent << Circuit::opPrefix <<
          "connect " << Circuit::varPrefix << buf2 << ", "
          << Circuit::varPrefix << buf1;
      out <<  " : " << Circuit::typePrefix << pair.second.type.name;
      out << "<" << pair.second.type.element_width << ">";
      out << ", ";
      out << Circuit::typePrefix << pair.second.type.name;
      out << "<" << pair.second.type.element_width << ">";
      out << "\n";
    }
  }
}

void FirrtlModule::printDeclaration(std::ostream &out) const {
  out << Circuit::indent << Circuit::opPrefix << "module @" <<
      name << " (\n";
  for (const auto &input : inputs) {
    out << Circuit::indent << Circuit::indent <<  "in " <<
        Circuit::varPrefix << input.name << " : " << Circuit::typePrefix <<
        input.type.name;
    if (input.type.element_width != 1) {
      out << "<" << input.type.element_width << ">,\n";
    } else {
      out << ",\n";
    }
  }
  bool hasComma = false;
  for (const auto &output : outputs) {
    out << (hasComma ? ",\n" : "\n");
    out << Circuit::indent << Circuit::indent <<  "out " <<
        Circuit::varPrefix << output.name << ": " << Circuit::typePrefix <<
        output.type.name;
    if (output.type.element_width != 1) {
      out << "<" << output.type.element_width << ">";
    }
    hasComma = true;
  }
  out << ")\n";
  out << Circuit::indent << Circuit::indent << "{\n";
}

void FirrtlModule::printEpilogue(std::ostream &out) const {
  out << Circuit::indent << Circuit::indent << "} ";
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
  	hackFirrtl(buf);
    out << Circuit::indent << buf;
    /*if (input.type.element_width != 1) {
      out << "[" << input.type.element_width - 1 << ":" << 0 << "]";
    }*/
    hasComma = true;
  }
  for (const auto &output : outputs) {
    out << (hasComma ? ",\n" : "\n");
	  buf.assign(output.name);
    hackFirrtl(buf);
    out << Circuit::indent << buf;
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
  out << Circuit::indent << Circuit::opPrefix << "extmodule @" <<
  name << "(";
  bool hasComma = false;
  std::string buf;
  for (const auto &input : inputs) {
	  buf.assign(input.name);
	  hackFirrtl(buf);
    out << (hasComma ? ",\n" : "\n");
    out << Circuit::indent << Circuit::indent <<  "in " <<
        buf << " : " << Circuit::typePrefix << input.type.name;
    if (input.type.element_width != 1) {
      out << "<" << input.type.element_width << ">";
    }
    hasComma = true;
  }
  for (const auto &output : outputs) {
	  buf.assign(output.name);
    hackFirrtl(buf);
    out << (hasComma ? ",\n" : "\n");
    out << Circuit::indent << Circuit::indent <<  "out " <<  buf <<
        ": " << Circuit::typePrefix << output.type.name;
    if (output.type.element_width != 1) {
      out << "<" << output.type.element_width << ">";
    }
  }
  out << ")\n";
}

void ExternalModule::moveVerilogModule(
    const std::string &outputDirName) const {
  std::string outputFileName = outputDirName + getFileNameFromPath(path);
  std::cout << toLower(path) << std::endl;
  /*std::filesystem::copy(toLower(path),
                        outputFileName,
                        std::filesystem::copy_options::overwrite_existing);*/
}

void ExternalModule::printVerilogModule(std::ostream &out) const {
  printDeclaration(out);
  printBody(out);
  printEpilogue(out);
}

void Circuit::printFirrtlModule(const FirrtlModule &firmodule,
                                std::ostream &out) const {
  firmodule.printFirrtlModule(out);
}

void Circuit::printDeclaration(std::ostream &out) const {
  out << Circuit::opPrefix << "circuit " << "\"" << name <<
      "\" " << "{\n";
}

void Circuit::printEpilogue(std::ostream &out) const {
  out << "}";
}

void Circuit::printFirrtlModules(std::ostream &out) const {
  for (const auto &pair : firModules) {
    pair.second.printFirrtlModule(out);
  }
}

void Circuit::printExtDeclarations(std::ostream &out) const {
  for (const auto &pair : extModules) {
    pair.second.printFirrtlDeclaration(out);
  }
}

void Circuit::printFirrtl(std::ostream &out) const {
  printDeclaration(out);
  printFirrtlModules(out);
  printExtDeclarations(out);
  printEpilogue(out);
}

void Circuit::moveVerilogLibrary(const std::string &outputDirName,
                                 std::ostream &out) const {
  for (const auto &pair : extModules) {
    if (pair.second.body == "") {
      pair.second.moveVerilogModule(outputDirName);
    } else {
      pair.second.printVerilogModule(out);
    }
  }
}

void Circuit::convertToSV(const std::string& inputFirrtlName) const {
  assert((system((std::string(Circuit::circt) +
         inputFirrtlName +
         std::string(Circuit::circt_options)).c_str()) != -1) &&
         "Error while creating top verilog module!");
}

void Circuit::printFiles(const std::string& outputFirrtlName,
                         const std::string& outputVerilogName,
                         const std::string& outputDirName) const {
  createDirRecur(outputDirName);
  std::ofstream outputFile;
  outputFile.open(outputDirName + outputFirrtlName);
  printFirrtl(outputFile);
  outputFile.close();
  convertToSV(outputDirName + outputFirrtlName);
  std::filesystem::rename("main.sv", (outputDirName +
                                      std::string("main.sv")).c_str());
  outputFile.open(outputDirName + outputVerilogName);
  moveVerilogLibrary(outputDirName, outputFile);
  outputFile.close();
}

std::shared_ptr<Circuit> Compiler::constructCircuit(
    const Model &model,
    const std::string& name) {
  auto circuit = std::make_shared<Circuit>(name);

  for (const auto *nodetype : model.nodetypes) {
    circuit->addExternalModule(nodetype);
  }
  circuit->addFirModule(FirrtlModule(model, name));

  return circuit;
}

void Circuit::printRndVlogTest(const Model &model,
                               const std::string &tstPath,
                               const int latency,
                               const size_t tstCnt) {

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
