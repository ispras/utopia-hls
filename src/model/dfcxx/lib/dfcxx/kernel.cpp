//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/converter.h"
#include "dfcxx/IRbuilders/builder.h"
#include "dfcxx/kernel.h"
#include "dfcxx/vars/constant.h"

#include "ctemplate/template.h"
#include "llvm/Support/raw_ostream.h"

#include <ctime>
#include <iostream>
#include <unordered_map>
#include <string>

namespace dfcxx {

Kernel::Kernel() : storage(), typeBuilder(), varBuilder(),
                   graph(), io(graph, typeBuilder, varBuilder, storage),
                   offset(graph, typeBuilder, varBuilder, storage),
                   constant(graph, typeBuilder, varBuilder, storage),
                   control(graph, typeBuilder, varBuilder, storage) {}

DFType Kernel::dfUInt(uint8_t bits) {
  DFTypeImpl *type = typeBuilder.buildFixed(SignMode::UNSIGNED, bits, 0);
  return DFType(storage.addType(type));
}

DFType Kernel::dfInt(uint8_t bits) {
  DFTypeImpl *type = typeBuilder.buildFixed(SignMode::SIGNED, bits, 0);
  return DFType(storage.addType(type));
}

DFType Kernel::dfFloat(uint8_t expBits, uint8_t fracBits) {
  DFTypeImpl *type = typeBuilder.buildFloat(expBits, fracBits);
  return DFType(storage.addType(type));
}

DFType Kernel::dfBool() {
  DFTypeImpl *type = typeBuilder.buildBool();
  return DFType(storage.addType(type));
}

bool Kernel::compileDot(llvm::raw_fd_ostream *stream) {
  using ctemplate::TemplateDictionary;
  
  uint64_t counter = 0;
  std::unordered_map<Node, std::string> idMapping;
  auto getName = [&idMapping, &counter] (const Node &node) -> std::string {
    // If the node is named - just return the name.
    auto name = DFVariable(node.var).getName();
    if (!name.empty()) {
      return name.data();
    }
    // If the mapping contains the node name - return it.
    auto it = idMapping.find(node);
    if (it != idMapping.end()) {
      return it->second;
    }
    // Otherwise create and return the new node name mapping. 
    return (idMapping[node] = "node" + std::to_string(counter++));
  };

  std::string result;
  TemplateDictionary *dict = new TemplateDictionary("dot");
  dict->SetValue("KERNEL_NAME", this->getName().data());
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

  for (Node node : graph.nodes) {
    TemplateDictionary *elem = dict->AddSectionDictionary("ELEMENTS");
    auto name = getName(node);
    elem->SetValue("NAME", name);
    std::string shape = "box";
    std::string label = name;
    auto data = node.data;
    switch (node.type) {
      case OFFSET:
        shape = "diamond";
        label = std::to_string(data.offset);
        break;
      case IN: shape = "invtriangle"; break;
      case OUT: shape = "triangle"; break;
      case MUX: shape = "invtrapezium"; break;
      case ADD: label = "+"; break;
      case SUB: label = "-"; break;
      case MUL: label = "*"; break;
      case DIV: label = "/"; break;
      case AND: label = "&"; break;
      case OR: label = "|"; break;
      case XOR: label = "^"; break;
      case NOT: label = "!"; break;
      case NEG: label = "-"; break;
      case LESS: label = "<"; break;
      case LESSEQ: label = "<="; break;
      case GREATER: label = ">"; break;
      case GREATEREQ: label = ">="; break;
      case EQ: label = "=="; break;
      case NEQ: label = "!="; break;
      case SHL: label = "<< " + std::to_string(data.bitShift); break;
      case SHR: label = ">> " + std::to_string(data.bitShift); break;
      default: break; // Silences -Wswitch warning for "CONST".
    }

    elem->SetValue("SHAPE", shape);
    elem->SetValue("LABEL", label);

    unsigned i = 0;
    for (Channel chan : graph.inputs[node]) {
      TemplateDictionary *conn = elem->AddSectionDictionary("CONNECTIONS");
      conn->SetValue("SRC_NAME", getName(chan.source));
      conn->SetValue("TRG_NAME", name);
      conn->SetValue("CON_LABEL", std::to_string(i++));
    }
  }

  ctemplate::ExpandTemplate(DOT_TEMPLATE_PATH, ctemplate::DO_NOT_STRIP,
                            dict, &result);
  delete dict;

  *stream << result;
  return true;
}

bool Kernel::compile(const DFLatencyConfig &config,
                     const std::vector<std::string> &outputPaths,
                     const Scheduler &sched) {
  DFCIRBuilder builder(config);
  auto compiled = builder.buildModule(this);
  size_t count = outputPaths.size();
  std::vector<llvm::raw_fd_ostream *> outputStreams(count);
  // Output paths strings are converted into output streams.
  for (unsigned i = 0; i < count; ++i) {
    std::error_code ec;
    outputStreams[i] = (!outputPaths[i].empty())
        ? new llvm::raw_fd_ostream(outputPaths[i], ec)
        : nullptr;
  }
  bool result = true;
  // Compile the kernel to DOT if such stream is specified.
  if (auto *stream = outputStreams[OUT_FORMAT_ID_INT(DOT)]) {
    result &= compileDot(stream);
  }
  if (result) {
    result &= DFCIRConverter(config).convertAndPrint(compiled,
                                                     outputStreams,
                                                     sched);
  }
  // Every created output stream has to be closed explicitly.
  for (llvm::raw_fd_ostream *stream : outputStreams) {
    if (stream) {
      stream->close();
      delete stream;
    }
  }
  return result;
}

bool Kernel::compile(const DFLatencyConfig &config,
                     const DFOutputPaths &outputPaths,
                     const Scheduler &sched) {
  std::vector<std::string> outPathsStrings(OUT_FORMAT_ID_INT(COUNT), "");
  for (const auto &[id, path] : outputPaths) {
    outPathsStrings[static_cast<uint8_t>(id)] = path;
  }
  return compile(config, outPathsStrings, sched);
}

} // namespace dfcxx
