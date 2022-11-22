//===- Dumper.cpp - HIL-to-mlir printer ---------------*- C++ -*-----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"

#include <cassert>
#include <fstream>
#include <iomanip>

#include "HIL/Dialect.h"
#include "HIL/Dumper.h"
#include "HIL/Ops.h"
#include "hls/model/model.h"
#include "hls/parser/hil/builder.h"

using eda::hls::model::Chan;
using eda::hls::model::Graph;
using eda::hls::model::Model;
using eda::hls::model::Node;
using eda::hls::model::NodeType;
using eda::hls::model::Port;
using eda::hls::model::Binding;

namespace detail {

template <typename T> class quoted {
public:
  quoted(const T &data) : data_(data) {}
  template <typename U>
  friend std::ostream &operator<<(std::ostream &, const quoted<U> &q);

private:
  const T &data_;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const quoted<T> &q) {
  return os << '"' << q.data_ << '"';
}

template <typename T> class curly_braced {
public:
  curly_braced(const T &data) : data_(data) {}
  template <typename U>
  friend std::ostream &operator<<(std::ostream &, const curly_braced<U> &q);

private:
  const T &data_;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const curly_braced<T> &q) {
  return os << '<' << q.data_ << '>';
}

class indented_ostream {
public:
  indented_ostream(std::ostream &os, size_t indent_step, char indent_char)
      : os_(os), indent_char_(indent_char), indent_step_(indent_step) {}

  void increase_indent() { indent_width_ += indent_step_; }

  void decrease_indent() {
    assert(indent_width_ >= indent_step_);
    indent_width_ -= indent_step_;
  }

  template <typename T> indented_ostream &operator<<(const T &x) {
    if (is_line_beginning_) {
      os_ << std::string(indent_width_, indent_char_);
      is_line_beginning_ = false;
    }
    os_ << x;
    return *this;
  }

  indented_ostream &operator<<(const char &x) {
    if (is_line_beginning_) {
      os_ << std::string(indent_width_, indent_char_);
      is_line_beginning_ = false;
    }
    os_ << x;
    if (x == '\n') {
      is_line_beginning_ = true;
    }
    return *this;
  }

  indented_ostream &operator<<(const char *const &s) {
    if (is_line_beginning_) {
      os_ << std::string(indent_width_, indent_char_);
      is_line_beginning_ = false;
    }
    std::string str(s);
    os_ << s;
    if (!str.empty() && str.back() == '\n') {
      is_line_beginning_ = true;
    }
    return *this;
  }

private:
  std::ostream &os_;
  std::size_t indent_width_ = 0;
  bool is_line_beginning_ = true;
  const char indent_char_;
  const std::size_t indent_step_;
};

class indent_block {
public:
  indent_block(indented_ostream &os) : os_(os) { os_.increase_indent(); }
  ~indent_block() { os_.decrease_indent(); }

private:
  indented_ostream &os_;
};

template <typename T> class ModelDumper {
public:
  ModelDumper(const T &node, indented_ostream &os) : node_(node), os_(os) {}
  void dump();
  template <typename U>
  static ModelDumper<U> get(const U &node, indented_ostream &os) {
    return ModelDumper<U>(node, os);
  }

private:
  const T &node_;
  indented_ostream &os_;
};

template <> void ModelDumper<Port>::dump() {
  os_ << "#hil.port<" << quoted(node_.name) << ' ' << quoted(node_.type.name)
      << ' ' << curly_braced(node_.flow) << ' ' << node_.latency << ' '
      << node_.isConst << ' ' << node_.value << '>';
}

template <> void ModelDumper<Binding>::dump() {
  os_ << "#hil.bnd<" << quoted((*(node_.node)).name) << ' ';
  ModelDumper::get(*(node_.port), os_).dump();
  os_ << '>';
}

template <> void ModelDumper<NodeType>::dump() {
  os_ << "hil.nodetype " << quoted(node_.name) << " [";
  {
    indent_block _(os_);
    bool print_sep = false;
    for (auto *input_arg : node_.inputs) {
      if (print_sep) {
        os_ << ", ";
      } else {
        os_ << '\n';
        print_sep = true;
      }
      ModelDumper::get(*input_arg, os_).dump();
    }
  }
  os_ << '\n' << "] => [";
  {
    indent_block _(os_);
    bool print_sep = false;
    for (auto *output_arg : node_.outputs) {
      if (print_sep) {
        os_ << ", ";
      } else {
        os_ << '\n';
      }
      print_sep = true;
      ModelDumper::get(*output_arg, os_).dump();
    }
  }
  os_ << '\n' << "]";
}

template <> void ModelDumper<Chan>::dump() {
  os_ << "hil.chan " << quoted(node_.type) << ' ' << quoted(node_.name) << ' ';
  ModelDumper::get(node_.source, os_).dump();
  os_ << " == ";
  ModelDumper::get(node_.target, os_).dump();
}

template <> void ModelDumper<Node>::dump() {
  os_ << "hil.node " << quoted(node_.type.name) << ' ' << quoted(node_.name)
      << " [";
  {
    indent_block _(os_);
    bool print_sep = false;
    for (auto *input_chan : node_.inputs) {
      if (print_sep) {
        os_ << ", ";
      } else {
        os_ << '\n';
        print_sep = true;
      }
      os_ << quoted(input_chan->name);
    }
  }
  os_ << '\n' << "] => [";
  {
    indent_block _(os_);
    bool print_sep = false;
    for (auto *output_chan : node_.outputs) {
      if (print_sep) {
        os_ << ", ";
      } else {
        os_ << '\n';
        print_sep = true;
      }
      os_ << quoted(output_chan->name);
    }
  }
  os_ << '\n' << "]";
}

template <> void ModelDumper<Graph>::dump() {
  os_ << "hil.graph " << quoted(node_.name) << " {\n";
  {
    indent_block _(os_);
    os_ << "hil.chans {\n";
    {
      indent_block _(os_);
      for (auto *chan : node_.chans) {
        ModelDumper::get(*chan, os_).dump();
        os_ << '\n';
      }
    }
    os_ << "}\n";
    os_ << "hil.nodes {\n";
    {
      indent_block _(os_);
      for (auto *node : node_.nodes) {
        ModelDumper::get(*node, os_).dump();
        os_ << '\n';
      }
    }
    os_ << "}\n";
  }
  os_ << "}\n";
}

template <> void ModelDumper<Model>::dump() {
  os_ << "hil.model " << quoted(node_.name) << " {" << '\n';
  {
    indent_block _(os_);
    os_ << "hil.nodetypes {\n";
    {
      indent_block _(os_);
      for (auto nodeTypeIterator = node_.nodetypes.begin();
           nodeTypeIterator != node_.nodetypes.end();
           nodeTypeIterator++) {
        ModelDumper::get(*(nodeTypeIterator->second), os_).dump();
        os_ << '\n';
      }
    }
    os_ << "}\n";
    for (auto *graph : node_.graphs) {
      ModelDumper::get(*graph, os_).dump();
    }
  }
  os_ << "}\n";
}

} // namespace detail

namespace eda::hls::model {
std::ostream &dump_model_mlir(const eda::hls::model::Model &model,
                              std::ostream &os) {
  os << std::fixed << std::setprecision(8);
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
