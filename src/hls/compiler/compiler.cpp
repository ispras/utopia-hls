//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <memory>

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

  ExternalModule::ExternalModule(const model::NodeType* nodetype) {
    externalModuleName = nodetype->name;
    addPort(Port("clock", true, 1, "clock"), true);
    addPort(Port("reset", true, 1, "reset"), true);
    for (const auto *input : nodetype->inputs) {
      addPort(Port(input->name, true), true);
    }
    for (const auto *output : nodetype->outputs) {
      addPort(Port(output->name, false), false);
    }
    auto meta = Library::get().find(*nodetype);
    auto element = meta->construct(meta->params);
    addBody(element->ir);
  }

  void ExternalModule::addBody(const std::string &body) {
    this->body = body;
  }

  void ExternalModule::addPort(const Port &Port, const bool isInput) {
    isInput ? inputs.push_back(Port) : outputs.push_back(Port);
  }

  void ExternalModule::printDeclaration(std::ostream &out) const {
    out << "module " << externalModuleName << "(\n";
    for (const auto &input : inputs) {
      out << Compiler::indent << input.portName;
       if (input.portWidth != 1) {
         out << "[" << input.portWidth << ":" << 0 << "],\n";
       } else {
         out << ",\n";
       }
    }
    bool hasComma = false;
    for (const auto &output : outputs) {
      out << (hasComma ? ",\n" : "\n");
      out << Compiler::indent << output.portName;
      if (output.portWidth != 1) {
       out << "[" << output.portWidth << ":" << 0 << "],\n";
      }
      hasComma = true;
    }
    out << ");\n";
  }

  void ExternalModule::printBody(std::ostream &out) const {
    out << body;
  }

  void ExternalModule::printEpilogue(std::ostream &out) const {
    out << "endmodule " << "//" << externalModuleName;
    printEmptyLine(out);
  }

  void ExternalModule::printEmptyLine(std::ostream &out) const {
    out << "\n";
  }

  void ExternalModule::printVerilog(std::ostream &out) const {
    printDeclaration(out);
    printBody(out);
    printEpilogue(out);
    printEmptyLine(out);
  }

  void ExternalModule::printFirrtlDeclaration(std::ostream &out) const {
    out << Compiler::indent << Compiler::opPrefix << "extmodule @" <<
     externalModuleName << "(";
     bool hasComma = false;
     for (const auto &input : inputs) {
       out << (hasComma ? ",\n" : "\n");
       out << Compiler::indent << Compiler::indent <<  "in " <<
        Compiler::varPrefix << input.portName << " : " << Compiler::typePrefix
        << input.portType;
        if (input.portType != "reset" && input.portType != "clock") {
          out << "<" << input.portWidth << ">";
        }
        hasComma = true;
     }
     for (const auto &output : outputs) {
       out << (hasComma ? ",\n" : "\n");
       out << Compiler::indent << Compiler::indent <<  "out " <<
       Compiler::varPrefix << output.portName << ": " << Compiler::typePrefix <<
        output.portType << "<" << output.portWidth << ">";
     }
     out << ")\n";
  }

  Module::Module(const eda::hls::model::Model &model) {
    const auto* graph = model.main();
    moduleName = graph->name;
    //Inputs & Outputs
    addPort(Port("clock", true, 1, "clock"), true);
    addPort(Port("reset", true, 1, "reset"), true);
    for (const auto *node : graph->nodes) {
      if (node->isSource()) {
        for (const auto *output : node->outputs) {
          addPort(Port(output->name, true), true);
        }
      }
      if (node->isSink()) {
        for (const auto *input : node->inputs) {
          addPort(Port(input->name, false), false);
        }
      }
    }
    //Instances
    for (const auto *node : graph->nodes) {
      addInstance(Instance(node->name,
                           node->type.name));
      for (const auto *input : node->inputs) {
        Port inputPort(node->name + "_" + input->target.port->name, true);
        instances[instances.size() - 1].addInput(inputPort);
      }

      for (const auto *output : node->outputs) {
        Port outputPort(node->name + "_" + output->source.port->name, false);
        instances[instances.size() - 1].addOutput(outputPort);
        if (node->type.name != "sink") {
          Port connectsTo(output->target.node->name + "_" +
           output->target.port->name, true);
          instances[instances.size() - 1].addBinding(outputPort, connectsTo);
        }
      }
      addBody("");
    }
  }

  void Module::addBody(const std::string &body) {
    this->body = body;
  }

  void Module::addWire(const Wire &inputWire) {
    wires.push_back(inputWire);
  }

  void Module::addInstance(const Instance &inputInstance) {
    instances.push_back(inputInstance);
  }

  void Module::addPort(const Port &Port, const bool isInput) {
    isInput ? inputs.push_back(Port) : outputs.push_back(Port);
  }

  void Module::printWires(std::ostream &out) const {
    for (const auto &wire : wires) {
      out << Compiler::indent << Compiler::indent << Compiler::varPrefix <<
      wire.wireName << " = " << Compiler::opPrefix << "wire :" <<
       Compiler::typePrefix << wire.wireType;
       if (wire.wireType != "clock" && wire.wireType != "reset") {
         out << "<" << wire.wireWidth << ">";
       }
       out << "\n";
    }
  }

  void Module::printInstances(std::ostream &out) const {
    for (const auto &instance : instances) {
      out << Compiler::indent << Compiler::indent;
      out << Compiler::varPrefix << instance.instanceName << "_" << "clock" <<
       "," << " ";
      out << Compiler::varPrefix << instance.instanceName << "_" << "reset" <<
       "," << " ";
      bool hasComma = false;
      for (const auto &input : instance.inputs) {
        out << (hasComma ? ", " : "");
        out << Compiler::varPrefix << input.portName;
        hasComma = true;
      }

      for (const auto &output : instance.outputs) {
        out << (hasComma ? ", " : "");
        out << Compiler::varPrefix << output.portName;
        hasComma = true;
      }

      out << " = " << Compiler::opPrefix << "instance @" << instance.moduleName
       << " {" << "name = " <<  "\"" << instance.instanceName << "\"}" << " : ";
      hasComma = false;
      out << Compiler::typePrefix << "clock, ";
      out << Compiler::typePrefix << "reset,";
      for (const auto &input : instance.inputs) {
        out << (hasComma ? ", " : " ");
        out << Compiler::typePrefix << input.portType;
        out << "<" << input.portWidth << ">";
        hasComma = true;
      }

      for (const auto &output : instance.outputs) {
        out << (hasComma ? ", " : " ");
        out << Compiler::typePrefix << output.portType;
        out << "<" << output.portWidth << ">";
        hasComma = true;
      }

      out << "\n";
    }
  }

  void Module::printConnections(std::ostream &out) const {
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
        "connect " << Compiler::varPrefix << pair.second.portName << ", "
         << Compiler::varPrefix << pair.first.portName;
         out <<  " : " << Compiler::typePrefix << pair.second.portType;
         out << "<" << pair.second.portWidth << ">";
         out << ", ";
         out << Compiler::typePrefix << pair.second.portType;
         out << "<" << pair.second.portWidth << ">";
         out << "\n";
      }
    }
  }

  void Module::printDeclaration(std::ostream &out) const {
    out << Compiler::indent << Compiler::opPrefix << "module @" << moduleName <<
     " (\n";
    for (const auto &input : inputs) {
      out << Compiler::indent << Compiler::indent <<  "in " <<
      Compiler::varPrefix << input.portName << " : " << Compiler::typePrefix <<
       input.portType;
       if (input.portName != "clock" && input.portName != "reset") {
         out << "<" << input.portWidth << ">,\n";
       } else {
         out << ",\n";
       }
    }
    bool hasComma = false;
    for (const auto &output : outputs) {
      out << (hasComma ? ",\n" : "\n");
      out << Compiler::indent << Compiler::indent <<  "out " <<
       Compiler::varPrefix << output.portName << ": " << Compiler::typePrefix <<
       output.portType << "<" << output.portWidth << ">";
      hasComma = true;
    }
    out << ")\n";
    out << Compiler::indent << Compiler::indent << "{\n";
  }

  void Module::printBody(std::ostream &out) const {
    out << body;
  }

  void Module::printEpilogue(std::ostream &out) const {
    out << Compiler::indent << Compiler::indent << "} ";
    printEmptyLine(out);
  }

  void Module::printEmptyLine(std::ostream &out) const {
    out << "\n";
  }

  void Module::printFirrtl(std::ostream &out) const {
    printDeclaration(out);
    printEmptyLine(out);
    printWires(out);
    printEmptyLine(out);
    printInstances(out);
    printConnections(out);
    printBody(out);
    printEpilogue(out);
    printEmptyLine(out);
  }

  Circuit::Circuit(std::string moduleName) : circuitName(moduleName) {}

  void Circuit::addModule(const Module &module) {
    modules.insert({module.moduleName, module});
  }

  void Circuit::addExternalModule(const ExternalModule &externalModule) {
    externalModules.insert({externalModule.externalModuleName, externalModule});
  }

  void Circuit::printFirrtl(std::ostream &out) const {
    out << Compiler::opPrefix << "circuit " << "\"" << circuitName <<
     "\" " << "{\n";
    for (const auto &pair : modules) {
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

  void Circuit::printFiles(const std::string& outputFirrtlName,
                           const std::string& outputVerilogName) const {
    std::ofstream outputFile;
    outputFile.open(outputFirrtlName);
    printFirrtl(outputFile);
    outputFile.close();
    outputFile.open(outputVerilogName);
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
  circuit->addModule(model);
  return circuit;
}

Compiler::Compiler(const eda::hls::model::Model &model) : model(model) {}

} // namespace eda::hls::compiler
