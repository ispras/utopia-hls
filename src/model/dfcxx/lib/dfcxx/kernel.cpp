//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/dfcir_builder.h"
#include "dfcxx/dfcir_processor.h"
#include "dfcxx/kernel.h"
#include "dfcxx/simulator.h"
#include "dfcxx/utils.h"
#include "dfcxx/vars/constant.h"

#include "ctemplate/template.h"
#include "llvm/Support/raw_ostream.h"

#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace dfcxx {

std::vector<Kernel *> Kernel::kernelStack;

Kernel::Kernel() : meta(), io(meta), offset(meta),
                   constant(meta), control(meta) {
  Kernel::kernelStack.push_back(this);
  if (Kernel::kernelStack.size() == 1) {
    KernelMeta::top = &meta;
  }
}

DFType Kernel::dfUInt(uint16_t bits) {
  auto *type = meta.typeBuilder.buildFixed(FixedType::SignMode::UNSIGNED,
                                           bits,
                                           0);
  return Kernel::getTopMeta()->storage.addType(type);
}

DFType Kernel::dfInt(uint16_t bits) {
  assert(bits > 0 && "Number of bits has to be greater than zero (for sign bit to exist).");
  auto *type = meta.typeBuilder.buildFixed(FixedType::SignMode::SIGNED,
                                           bits - 1,
                                           0);
  return Kernel::getTopMeta()->storage.addType(type);
}

DFType Kernel::dfFloat(uint16_t expBits, uint16_t fracBits) {
  auto *type = meta.typeBuilder.buildFloat(expBits, fracBits);
  return Kernel::getTopMeta()->storage.addType(type);
}

DFType Kernel::dfRawBits(uint16_t bits) {
  auto *type = meta.typeBuilder.buildRawBits(bits);
  return Kernel::getTopMeta()->storage.addType(type);
}

DFType Kernel::dfBool() {
  auto *type = meta.typeBuilder.buildBool();
  return Kernel::getTopMeta()->storage.addType(type);
}

const Graph &Kernel::getGraph() const {
  return meta.graph;
}

bool Kernel::compileDot(llvm::raw_fd_ostream *stream) {
  using ctemplate::TemplateDictionary;

  uint64_t counter = 0;
  std::unordered_map<const Node *, std::string> idMapping;
  auto getName = [&idMapping, &counter] (const Node *node) -> std::string {
    // If the node is named - just return the name.
    auto name = DFVariable(node->var).getName();
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

  const auto &nodes = meta.graph.getNodes();

  for (Node *node : nodes) {
    TemplateDictionary *elem = dict->AddSectionDictionary("ELEMENTS");
    auto name = getName(node);
    elem->SetValue("NAME", name);
    std::string shape = "box";
    std::string label = name;
    auto data = node->data;
    switch (node->type) {
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
    for (Channel *chan : node->inputs) {
      TemplateDictionary *conn = elem->AddSectionDictionary("CONNECTIONS");
      conn->SetValue("SRC_NAME", getName(chan->source));
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

void Kernel::rebindInput(DFVariable source, Node *input, Kernel &kern) {
  Node *sourceNode = meta.graph.findNode(source);
  meta.graph.rebindInput(sourceNode, input);

  kern.deleteNode(input);
}

DFVariable Kernel::rebindOutput(Node *output, DFVariable target, Kernel &kern) {
  Node *targetNode = meta.graph.findNode(target);
  Node *node = meta.graph.rebindOutput(output, targetNode);

  if (targetNode != node) {
    deleteNode(targetNode);
  }

  kern.deleteNode(output);
  return node->var;
}

void Kernel::deleteNode(Node *node) {
  meta.storage.deleteVariable(node->var);
  meta.graph.deleteNode(node);
}

bool Kernel::compile(const DFLatencyConfig &config,
                     const std::vector<std::string> &outputPaths,
                     const DFOptionsConfig &options) {
  DFCIRBuilder builder;
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
    result &= DFCIRProcessor(config).convertAndPrint(compiled,
                                                     outputStreams,
                                                     options);
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
                     const DFOptionsConfig &options) {
  return compile(config, outputPathsToVector(outputPaths), options);
}

bool Kernel::compile(const DFLatencyConfig &config,
                     const DFOutputPaths &outputPaths,
                     const Scheduler &sched) {
  DFOptionsConfig options;
  options.scheduler = sched;
  return compile(config, outputPaths, options);
}


bool Kernel::simulate(const std::string &inDataPath,
                      const std::string &outFilePath) {
  std::vector<Node *> sorted = topSort(meta.graph);
  DFCXXSimulator sim(sorted);
  std::ifstream input(inDataPath, std::ios::in);
  if (!input || input.bad() || input.eof() || input.fail() || !input.is_open()) {
    return false;
  }
  std::ofstream output(outFilePath, std::ios::out);
  return sim.simulate(input, output);
}

bool Kernel::check() const {
  const auto &nodes = meta.graph.getNodes();
  const auto &startNodes = meta.graph.getStartNodes();
  std::cout << "[UTOPIA] Kernel: " << getName() << std::endl;
  std::cout << "[UTOPIA] Nodes: " << nodes.size() << std::endl;
  std::cout << "[UTOPIA] Start/constant nodes: " << startNodes.size() << std::endl;

  return checkValidNodes();
}

bool Kernel::checkValidNodes() const {
  std::cout << "[UTOPIA] Checking whether constructed nodes are valid: ";

  const auto &nodes = meta.graph.getNodes();
  for (const Node *node: nodes) {
    if (node->type == OpType::NONE) {
      std::cout << "found invalid node(s). Abort." << std::endl;
      return false;
    }
  }
  std::cout << "finished." << std::endl;
  return true;
}

} // namespace dfcxx
