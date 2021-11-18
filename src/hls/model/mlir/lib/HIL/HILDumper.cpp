//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//
#include "mlir/InitAllDialects.h"

#include <cassert>
#include <fstream>

#include "HIL/HILDialect.h"
#include "HIL/HILDumper.h"
#include "HIL/HILOps.h"
#include "hls/model/model.h"
#include "hls/parser/hil/builder.h"

using eda::hls::model::Chan;
using eda::hls::model::Graph;
using eda::hls::model::Model;
using eda::hls::model::Node;
using eda::hls::model::NodeType;
using ModelPort = eda::hls::model::Port;

namespace detail {

template <typename T>
class quoted {
public:
  quoted(const T &data) : data(data) {}
  template <typename U>
  friend std::ostream &operator<<(std::ostream &, const quoted<U> &q);

private:
  const T &data;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const quoted<T> &q) {
  return os << '"' << q.data << '"';
}

template <typename T>
class curly_braced {
public:
  curly_braced(const T &data) : data(data) {}
  template <typename U>
  friend std::ostream &operator<<(std::ostream &, const curly_braced<U> &q);

private:
  const T &data;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const curly_braced<T> &q) {
  return os << '<' << q.data << '>';
}

class indented_ostream {
public:
  indented_ostream(std::ostream &os, size_t indent_step, char indent_char)
      : os(os), indent_char(indent_char), indent_step(indent_step) {}

  void increase_indent() { indent_width += indent_step; }

  void decrease_indent() {
    assert(indent_width >= indent_step);
    indent_width -= indent_step;
  }

  template <typename T>
  indented_ostream &operator<<(const T &x) {
    if (is_line_beginning) {
      os << std::string(indent_width, indent_char);
      is_line_beginning = false;
    }
    os << x;
    return *this;
  }

  indented_ostream &operator<<(const char &x) {
    if (is_line_beginning) {
      os << std::string(indent_width, indent_char);
      is_line_beginning = false;
    }
    os << x;
    if (x == '\n') {
      is_line_beginning = true;
    }
    return *this;
  }

  indented_ostream &operator<<(const char *const &s) {
    if (is_line_beginning) {
      os << std::string(indent_width, indent_char);
      is_line_beginning = false;
    }
    std::string str(s);
    os << s;
    if (!str.empty() && str.back() == '\n') {
      is_line_beginning = true;
    }
    return *this;
  }

private:
  std::ostream &os;
  std::size_t indent_width = 0;
  bool is_line_beginning = true;
  const char indent_char;
  const std::size_t indent_step;
};

class indent_block {
public:
  indent_block(indented_ostream &os) : os(os) { os.increase_indent(); }
  ~indent_block() { os.decrease_indent(); }

private:
  indented_ostream &os;
};

template <typename T>
class ModelDumper {
public:
  ModelDumper(const T &node, indented_ostream &os) : node(node), os(os) {}
  void dump();
  template <typename U>
  static ModelDumper<U> get(const U &node, indented_ostream &os) {
    return ModelDumper<U>(node, os);
  }

private:
  const T &node;
  indented_ostream &os;
};

enum PortType { Input, Output };
template <PortType>
struct Port {
  Port(const ModelPort &model_port) : model_port(model_port) {}
  const ModelPort &model_port;
};
using InputPort = Port<PortType::Input>;
using OutputPort = Port<PortType::Output>;

template <>
void ModelDumper<Port<Input>>::dump() {
  const auto &port = node.model_port;
  os << "!hil.input<" << quoted(port.type) << curly_braced(port.flow) << ' '
     << quoted(port.name) << ">";
}

template <>
void ModelDumper<Port<Output>>::dump() {
  const auto &port = node.model_port;
  os << "!hil.output<" << quoted(port.type) << curly_braced(port.flow) << ' '
     << port.latency << ' ' << quoted(port.name);
  if (port.isConst) {
    os << " = " << port.value;
  }
  os << ">";
}

template <>
void ModelDumper<NodeType>::dump() {
  os << "hil.nodetype " << quoted(node.name) << " [";
  {
    indent_block _(os);
    bool print_sep = false;
    for (auto input_arg : node.inputs) {
      if (print_sep) {
        os << ", ";
      } else {
        os << '\n';
        print_sep = true;
      }
      ModelDumper::get(InputPort(*input_arg), os).dump();
    }
  }
  os << '\n' << "] => [";
  {
    indent_block _(os);
    bool print_sep = false;
    for (auto output_arg : node.outputs) {
      if (print_sep) {
        os << ", ";
      } else {
        os << '\n';
        print_sep = true;
      }
      print_sep = true;
      ModelDumper::get(OutputPort(*output_arg), os).dump();
    }
  }
  os << '\n' << "]";
}

template <>
void ModelDumper<Chan>::dump() {
  os << "hil.chan " << quoted(node.type) << ' ' << quoted(node.name);
}

template <>
void ModelDumper<Node>::dump() {
  os << "hil.node " << quoted(node.type.name) << ' ' << quoted(node.name)
     << " [";
  {
    indent_block _(os);
    bool print_sep = false;
    for (auto input_chan : node.inputs) {
      if (print_sep) {
        os << ", ";
      } else {
        os << '\n';
        print_sep = true;
      }
      os << quoted(input_chan->name);
    }
  }
  os << '\n' << "] => [";
  {
    indent_block _(os);
    bool print_sep = false;
    for (auto output_chan : node.outputs) {
      if (print_sep) {
        os << ", ";
      } else {
        os << '\n';
        print_sep = true;
      }
      os << quoted(output_chan->name);
    }
  }
  os << '\n' << "]";
}

template <>
void ModelDumper<Graph>::dump() {
  os << "hil.graph " << quoted(node.name) << "{\n";
  {
    indent_block _(os);
    os << "hil.chans {\n";
    {
      indent_block _(os);
      for (auto chan : node.chans) {
        ModelDumper::get(*chan, os).dump();
        os << '\n';
      }
    }
    os << "}\n";
    os << "hil.nodes {\n";
    {
      indent_block _(os);
      for (auto node : node.nodes) {
        ModelDumper::get(*node, os).dump();
        os << '\n';
      }
    }
    os << "}\n";
  }
  os << "}\n";
}

template <>
void ModelDumper<Model>::dump() {
  os << "hil.model " << quoted(node.name) << " {" << '\n';
  {
    indent_block _(os);
    os << "hil.nodetypes {\n";
    {
      indent_block _(os);
      for (auto nodetype : node.nodetypes) {
        ModelDumper::get(*nodetype, os).dump();
        os << '\n';
      }
    }
    os << "}\n";
    for (auto graph : node.graphs) {
      ModelDumper::get(*graph, os).dump();
    }
  }
  os << "}\n";
}

} // namespace detail

namespace eda::hls::model {
std::ostream &dump_model_mlir(const eda::hls::model::Model &model,
                              std::ostream &os) {
  detail::indented_ostream ios(os, 2, ' ');
  detail::ModelDumper(model, ios).dump();
  return os;
}

void dump_model_mlir_to_file(const eda::hls::model::Model &model,
                             const std::string &filename) {
  std::ofstream os(filename);
  dump_model_mlir(model, os);
}
} // namespace eda::hls::model
