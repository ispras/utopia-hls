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
  const auto i = std::find_if(library.begin(), library.end(),
    [&name](const std::shared_ptr<MetaElement> &metaElement) {
      return metaElement->name == name;
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
  out << ");" << std::endl;

  auto meta = Library::get().find(type);
  // params.value["f"] can be adjusted before calling construct
  // params.set(std::string("f"), 3);
  auto element = meta->construct(meta->params);
  out << element->ir << std::endl;

  out << "endmodule" << " // " << type.name << std::endl;
}

void VerilogGraphPrinter::printChan(std::ostream &out, const eda::hls::model::Chan &chan) const {
  // TODO: chan.type is ignored
  out << "." << chan.source.node->name << "(" << chan.target.node->name << ")";
}

void VerilogGraphPrinter::print(std::ostream &out) const {
  out << "module " << graph.name << "();" << std::endl;

  for (const auto *chan : graph.chans) {
    out << "wire " << chan->name << ";" << std::endl;
  }
  for (const auto *node : graph.nodes) {
    // TODO: node.type is ignored
    out << node->type.name << " " << node->name << "(";

    bool comma = false;

    for (const auto *input : node->inputs) {
      out << (comma ? ", " : "");
      printChan(out, *input);
      comma = true;
    }

    for (const auto *output: node->outputs) {
      out << (comma ? ", " : "");
      printChan(out, *output);
      comma = true;
    }
    out << ");" << std::endl;
  }
  out << "endmodule // " << graph.name << std::endl;
}

std::ostream& operator <<(std::ostream &out, const VerilogNodeTypePrinter &printer) {
  printer.print(out);
  return out;
}

std::ostream& operator <<(std::ostream &out, const VerilogGraphPrinter &printer) {
  printer.print(out);
  return out;
}

} // namespace eda::hls::library
