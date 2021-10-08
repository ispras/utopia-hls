//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/compiler/compiler.h>
#include <hls/library/library.h>

namespace eda::hls::compiler {

// TODO: move this code into VerilogPrinter
void Compiler::printChan(std::ostream &out, const eda::hls::model::Chan *chan) const {
  // TODO: chan.type is ignored
  out << chan.name << " " < (chan->source != nullptr ? chan->source : "") <<
                            (chan->target != nullptr ? chan->target : "") << std::endl;
}

void Compiler::print(std::ostream &out) const {
  // print model.nodetypes
  for (const auto *nodetype : model->nodetypes) {
    auto printer = std::make_unique<VerilogPrinter>(*nodetype);
    std::out << *printer;
  }

  // print model.graphs TODO: refactor it by moving to VerilogPrinter!
  for (const auto *graph : model->graphs) {
    out << graph.name << "()" << std::endl;
    for (const auto *chan : graph.chans) {
      printChan(out, chan);
    }
    for (const auto *node : graph->nodes) {
      // TODO: node.type is ignored
      out << node.name << std::endl;
      for (const auto *input : node->inputs) {
        printChan(out, input);
      }
      for (const auto *output: node->outputs) {
        printChan(out, output);
      }
    }
  }
}

std::ostream& operator <<(std::ostream &out, const Compiler &compiler) {
  compiler.print(out);
  return out;
}

} // namespace eda::hls::compiler
