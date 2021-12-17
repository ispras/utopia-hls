//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/library.h"
#include "hls/library/library_mock.h"
#include "util/assert.h"

#include <algorithm>

namespace eda::hls::library {

std::shared_ptr<MetaElement> Library::find(const NodeType &nodetype) {
  auto metaElement = find(nodetype.name);

  if (metaElement == nullptr) {
    // FIXME
    metaElement = MetaElementMock::create(nodetype);
    library.push_back(metaElement);
  }

  return metaElement;
}

std::shared_ptr<MetaElement> Library::find(const std::string &name) const {
  std::string lowerCaseName = name;
  unsigned x = 0;
  while(name[x]) {
    lowerCaseName[x] = tolower(lowerCaseName[x]);
    x++;
  }

  const auto i = std::find_if(library.begin(), library.end(),
    [&lowerCaseName](const std::shared_ptr<MetaElement> &metaElement) {
      return metaElement->name == lowerCaseName;
    });

  return i == library.end() ? nullptr : *i;
}

void VerilogNodeTypePrinter::print(std::ostream &out) const {
  out << "module " << type.name << "(" << std::endl;

  bool comma = false;
  for (const auto *arg: type.inputs) {
    out << (comma ? ", " : "");
    out << arg->name;
    comma = true;
  }

  for (const auto *arg: type.outputs) {
    out << (comma ? ", " : "");
    out << arg->name;
    comma = true;
  }
  out << (comma ? ", " : "") << "clock, reset);" << std::endl;

  auto meta = Library::get().find(type);
  // params.value["f"] can be adjusted before calling construct
  // params.set(std::string("f"), 3);
  auto element = meta->construct(meta->params);
  out << element->ir << std::endl;

  out << "endmodule" << " // " << type.name << std::endl;
}

const std::string VerilogGraphPrinter::chanSourceToString(const eda::hls::model::Chan &chan) const {
  // TODO: chan.type is ignored
  return chan.source.node->name + "_" + chan.source.port->name;
}

void VerilogGraphPrinter::print(std::ostream &out) const {
  std::string top, wires, binds;

  top = std::string("module ") + graph.name + "();\n";
  wires += std::string("wire clock;\nwire reset;\n");

  for (const auto *node : graph.nodes) {
    // TODO: node.type is ignored
    binds += node->type.name + " " + node->name + "(";

    bool comma = false;

    for (const auto *input : node->inputs) {
      std::string chanName = chanSourceToString(*input);
      binds += std::string(comma ? ", " : "") + "." + input->target.port->name + "(" + chanName + ")";
      wires += std::string("wire ") + chanName + ";\n";
      comma = true;
    }

    for (const auto *output: node->outputs) {
      std::string chanName = chanSourceToString(*output);
      binds += std::string(comma ? ", " : "") + "." + output->source.port->name + "(" + chanName + ")";
      comma = true;
    }
    binds += std::string(comma ? ", " : "") + ".clock(clock), .reset(reset));\n";
  }
  out << top << wires << binds << "endmodule // " << graph.name << std::endl;
}

} // namespace eda::hls::library
