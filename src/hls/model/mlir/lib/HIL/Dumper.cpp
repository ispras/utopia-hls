//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//
// HIL-to-mlir printer.
//
//===----------------------------------------------------------------------===//

#include "HIL/Dumper.h"

#include "HIL/Dialect.h"
#include "HIL/Ops.h"
#include "hls/model/model.h"
#include "hls/parser/hil/builder.h"
#include "mlir/InitAllDialects.h"

#include <cassert>
#include <fstream>
#include <iomanip>

using eda::hls::model::Chan;
using eda::hls::model::Graph;
using eda::hls::model::Model;
using eda::hls::model::Node;
using eda::hls::model::NodeType;
using eda::hls::model::Port;
using eda::hls::model::Binding;

namespace detail {

template <typename T> class Quoted {
public:
  Quoted(const T &data) : data(data) {}
  template <typename U>
  friend std::ostream &operator<<(std::ostream &, const Quoted<U> &quoted);

private:
  const T &data;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Quoted<T> &quoted) {
  return os << '"' << quoted.data << '"';
}

template <typename T> class CurlyBraced {
public:
  CurlyBraced(const T &data) : data(data) {}
  template <typename U>
  friend std::ostream &operator<<(std::ostream &, const CurlyBraced<U> &curly);

private:
  const T &data;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const CurlyBraced<T> &curly) {
  return os << '<' << curly.data << '>';
}

class IndentedOstream {
public:
  IndentedOstream(std::ostream &os,
                  const std::size_t indentStep,
                  const char indentChar)
      : os(os), indentChar(indentChar), indentStep(indentStep) {}

  void increaseIndent() { indentWidth += indentStep; }

  void decreaseIndent() {
    assert(indentWidth >= indentStep);
    indentWidth -= indentStep;
  }

  template <typename T> IndentedOstream &operator<<(const T &x) {
    if (isLineBeginning) {
      os << std::string(indentWidth, indentChar);
      isLineBeginning = false;
    }
    os << x;
    return *this;
  }

  IndentedOstream &operator<<(const char &x) {
    if (isLineBeginning) {
      os << std::string(indentWidth, indentChar);
      isLineBeginning = false;
    }
    os << x;
    if (x == '\n') {
      isLineBeginning = true;
    }
    return *this;
  }

  IndentedOstream &operator<<(const char *const &s) {
    if (isLineBeginning) {
      os << std::string(indentWidth, indentChar);
      isLineBeginning = false;
    }
    std::string str(s);
    os << s;
    if (!str.empty() && str.back() == '\n') {
      isLineBeginning = true;
    }
    return *this;
  }

private:
  std::ostream &os;
  std::size_t indentWidth = 0;
  bool isLineBeginning = true;
  const char indentChar;
  const std::size_t indentStep;
};

class IndentBlock {
public:
  IndentBlock(IndentedOstream &os) : os(os) { os.increaseIndent(); }
  ~IndentBlock() { os.decreaseIndent(); }

private:
  IndentedOstream &os;
};

template <typename T> class ModelDumper {
public:
  ModelDumper(const T &node, IndentedOstream &os) : node(node), os(os) {}
  void dump();
  template <typename U>
  static ModelDumper<U> get(const U &node, IndentedOstream &os) {
    return ModelDumper<U>(node, os);
  }

private:
  const T &node;
  IndentedOstream &os;
};

template <> void ModelDumper<Port>::dump() {
  os << "#hil.port<" << Quoted(node.name) << ' ' << Quoted(node.type.name)
     << ' ' << CurlyBraced(node.flow) << ' ' << node.latency << ' '
     << node.isConst << ' ' << node.value.getIntValue() << '>';
}

template <> void ModelDumper<Binding>::dump() {
  os << "#hil.bnd<" << Quoted((*(node.node)).name) << ' ';
  ModelDumper::get(*(node.port), os).dump();
  os << '>';
}

template <> void ModelDumper<BindingGraph>::dump() {
  os << "#hil.bndgraph<" << Quoted((*(node.graph)).name) << ' '
     << Quoted((*(node.chan)).name) << ' ';
  os << '>';
}

template <> void ModelDumper<NodeType>::dump() {
  os << "hil.nodetype " << Quoted(node.name) << " [";
  {
    IndentBlock _(os);
    bool printSep = false;
    for (auto *inputArg : node.inputs) {
      if (printSep) {
        os << ", ";
      } else {
        os << '\n';
        printSep = true;
      }
      ModelDumper::get(*inputArg, os).dump();
    }
  }
  os << '\n' << "] => [";
  {
    IndentBlock _(os);
    bool printSep = false;
    for (auto *outputArg : node.outputs) {
      if (printSep) {
        os << ", ";
      } else {
        os << '\n';
      }
      printSep = true;
      ModelDumper::get(*outputArg, os).dump();
    }
  }
  os << '\n' << "]";
}

template <> void ModelDumper<Chan>::dump() {
  os << "hil.chan " << Quoted(node.type) << ' ' << Quoted(node.name) << ' ';
  ModelDumper::get(node.source, os).dump();
  os << " == ";
  ModelDumper::get(node.target, os).dump();
}

template <> void ModelDumper<Node>::dump() {
  os << "hil.node " << Quoted(node.type.name) << ' ' << Quoted(node.name)
     << " [";
  {
    IndentBlock _(os);
    bool printSep = false;
    for (auto *inputChan : node.inputs) {
      if (printSep) {
        os << ", ";
      } else {
        os << '\n';
        printSep = true;
      }
      os << Quoted(inputChan->name);
    }
  }
  os << '\n' << "] => [";
  {
    IndentBlock _(os);
    bool printSep = false;
    for (auto *outputChan : node.outputs) {
      if (printSep) {
        os << ", ";
      } else {
        os << '\n';
        printSep = true;
      }
      os << Quoted(outputChan->name);
    }
  }
  os << '\n' << "]";
}

template <> void ModelDumper<Con>::dump() {
  os << "hil.con" << " " << Quoted(node.type) << " " << Quoted(node.name) 
     << " " << Quoted(node.dir) << " ";
  ModelDumper::get(node.source, os).dump();
  os << " == ";
  ModelDumper::get(node.target, os).dump();
}

template <> void ModelDumper<Graph>::dump() {
  os << "hil.graph " << Quoted(node.name) << " {\n";
    IndentBlock _(os);
    os << "hil.chans {\n";
    {
      IndentBlock _(os);
      for (auto *chan : node.chans) {
        ModelDumper::get(*chan, os).dump();
        os << '\n';
      }
    }
    os << "}\n";
    os << "hil.nodes {\n";
    {
      IndentBlock _(os);
      for (auto *node : node.nodes) {
        ModelDumper::get(*node, os).dump();
        os << '\n';
      }
    }
    os << "}\n";
    os << "hil.insts {\n";
    {
      IndentBlock _(os);
      for (const auto &pair : node.cons) {
        os << "hil.inst " << Quoted(pair.first) << " {\n";
        {
          IndentBlock _(os);
          os << "hil.cons " << "{\n";
          {
            IndentBlock _(os);
            for (auto *con : pair.second) {
              ModelDumper::get(*con, os).dump();
              os << '\n';
          }
          os << "}\n"; // Cons close
          }
        os << "}\n"; // Inst close
        }
      }
    }
    os << "}\n"; // Insts close
  os << "}\n";// Graph close
}

template <> void ModelDumper<Model>::dump() {
  os << "hil.model " << Quoted(node.name) << " {" << '\n';
  {
    IndentBlock _(os);
    os << "hil.nodetypes {\n";
    {
      IndentBlock _(os);
      for (auto nodeTypeIterator = node.nodetypes.begin();
           nodeTypeIterator != node.nodetypes.end();
           nodeTypeIterator++) {
        ModelDumper::get(*(nodeTypeIterator->second), os).dump();
        os << '\n';
      }
    }
    os << "}\n";
    
    os << "hil.graphs {\n";
    {
      IndentBlock _(os);
      for (auto *graph : node.graphs) {
        ModelDumper::get(*graph, os).dump();
      }
    }
    os << "}\n";
  }
  os << "}\n";
}

} // namespace detail

namespace eda::hls::model {
std::ostream &dumpModelMlir(const eda::hls::model::Model &model,
                            std::ostream &os) {
  os << std::fixed << std::setprecision(8);
  detail::IndentedOstream ios(os, 2, ' ');
  detail::ModelDumper(model, ios).dump();
  return os;
}

void dumpModelMlirToFile(const eda::hls::model::Model &model,
                         const std::string &filename) {
  std::ofstream os(filename);
  dumpModelMlir(model, os);
}

} // namespace eda::hls::model