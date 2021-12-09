//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string.h>

#include "hls/compiler/compiler.h"
#include "hls/library/library.h"
#include "hls/library/library_mock.h"

using namespace eda::hls::library;

namespace eda::hls::compiler {

  const std::string chanSourceToString(const eda::hls::model::Chan &chan) {
    return chan.source.node->name + "_" + chan.source.port->name;
  }

  void Instance::addInput(const Port &inputPort) {
    inputs.push_back(inputPort);
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
    addBody(element->ir)  /*void Module::addWire(const Wire &inputWire) {
    wires.push_back(inputWire);
  }*/;
  }

  void ExternalModule::printDeclaration(std::ostream &out) const {
    out << "module " << moduleName << "(\n";
    for (const auto &input : inputs) {
      out << Compiler::indent << input.name;
      if (input.width != 1) {
        out << "[" << input.width << ":" << 0 << "],\n";
      } else {
        out << ",\n";
      }
    }
    bool hasComma = false;
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
          Compiler::varPrefix << input.name << " : " << Compiler::typePrefix
          << input.type;
      if (input.type != "reset" && input.type != "clock") {
        out << "<" << input.width << ">";
      }
      hasComma = true;
    }
    for (const auto &output : outputs) {
      out << (hasComma ? ",\n" : "\n");
      out << Compiler::indent << Compiler::indent <<  "out " <<
          Compiler::varPrefix << output.name << ": " << Compiler::typePrefix <<
          output.type << "<" << output.width << ">";
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
        instances.back().addInput(inputPort);  /*void Module::addWire(const Wire &inputWire) {
    wires.push_back(inputWire);
  }*/
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

      out << " = " << Compiler::opPrefix << "instance @" << instance.moduleName
          << " {" << "name = " <<  "\"" << instance.instanceName << "\"}" <<
          " : ";
      hasComma = false;
      out << Compiler::typePrefix << "clock, ";
      out << Compiler::typePrefix << "reset,";
      for (const auto &input : instance.inputs) {
        out << (hasComma ? ", " : " ");
        out << Compiler::typePrefix << input.type;
        out << "<" << input.width << ">";
        hasComma = true;
      }

      for (const auto &output : instance.outputs) {
        out << (hasComma ? ", " : " ");
        out << Compiler::typePrefix << output.type;
        out << "<" << output.width << ">";
        hasComma = true;
      }

      out << "\n";
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
    externalModules.insert({externalModule.moduleName, externalModule});
  }

  void Circuit::printFirrtl(std::ostream &out) const {
    out << Compiler::opPrefix << "circuit " << "\"" << name <<
        "\" " << "{\n";
    for (const auto &pair : firModules) {
      pair.second.printFirrtl(out);
    }
    for (const auto &pair : externalModules) {
      pair.second.printFirrtlDeclaration(out);
    }
    out << "}";
  }

  void Circuit::printVerilog(std::ostream &out) const {
    for (const auto &pair : externalModules) {
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
                           const std::string& testName) const {
    std::ofstream outputFile;
    outputFile.open(Compiler::relativePath + outputFirrtlName);
    printFirrtl(outputFile);
    outputFile.close();
    convertToSV(Compiler::relativePath + outputFirrtlName);
    std::filesystem::create_directory(Compiler::relativePath + testName);
    std::filesystem::rename("main.sv", (Compiler::relativePath +
                                        testName +
                                        "/" +
                                        std::string("main.sv")).c_str());
    outputFile.open(Compiler::relativePath + outputVerilogName);
    printVerilog(outputFile);
    outputFile.close();
  }

  std::ostream& operator <<(std::ostream &out, const Circuit &circuit) {
    circuit.printFirrtl(out);
    circuit.printVerilog(out);
    return out;
  }

std::shared_ptr<Circuit> Compiler::constructCircuit() {
  auto circuit = std::make_shared<Circuit>(std::string(model.main()->name));
  for (const auto *nodetype : model.nodetypes) {
    circuit->addExternalModule(nodetype);
  }
  circuit->addFirModule(model);
  return circuit;
}

Compiler::Compiler(const eda::hls::model::Model &model) : model(model) {}

} // namespace eda::hls::compiler
