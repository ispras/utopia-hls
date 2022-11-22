//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "hls/model/indicators.h"
#include "hls/model/parameters.h"
#include "util/string.h"

using namespace eda::utils;

namespace eda::hls::library {
struct MetaElement;
} // namespace eda::hls::library

namespace eda::hls::model {
struct NodeType;
} // namespace eda::hls::model

namespace eda::hls::model {
/// Key for MetaElement / Nodetype / Unit 
struct Signature {
  Signature(const std::string &name, 
            const std::vector<std::string> &inputTypeNames,
            const std::vector<std::string> &outputTypeNames);
  Signature(const eda::hls::model::NodeType &nodeType);
  bool operator==(const Signature &signature) const;
  std::string name;
  std::vector<std::string> inputTypeNames;
  std::vector<std::string> outputTypeNames;
};
} // namespace eda::hls::model

namespace std {
template<>
struct hash<eda::hls::model::Signature> {
  size_t operator()(const eda::hls::model::Signature &signature) const {
    size_t hash = std::hash<std::string>()(signature.name);
    for (const auto &inputTypeName : signature.inputTypeNames) {
      hash = hash * 13 + std::hash<std::string>()(inputTypeName);
    }
    for (const auto &outputTypeName : signature.outputTypeNames) {
      hash = hash * 13 + std::hash<std::string>()(outputTypeName);
    }
    return hash;
  }
};
} // namespace std

namespace eda::hls::model {

struct Graph;
struct Model;
struct Node;
struct Transform;


//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

struct Type {
  enum Kind {
    CUSTOM, // uninterpreted identifer
    FIXED,  // fixed<IntBits, FracBits, IsSigned>
    FLOAT,  // float<ExpBits, Precision>
    TUPLE,  // tuple<Type1, ..., TypeN>
    TENSOR  // tensor<Type, Dim1, ..., DimN>
  };

  static const Type& get(const std::string &name);

  Type(Kind kind, const std::string &name, std::size_t size):
    kind(kind), name(name), size(size) {}

  virtual ~Type() {}

  bool operator==(const Type &rhs) const {
    return name == rhs.name;
  }

  bool operator!=(const Type &rhs) const {
    return name != rhs.name;
  }

  const Kind kind;
  const std::string name;
  const std::size_t size;
};

struct CustomType final: public Type {
  CustomType(const std::string name):
    Type(CUSTOM, name, 1) {}
};

struct FixedType final: public Type {
  FixedType(std::size_t intBits, std::size_t fracBits, bool isSigned):
    Type(FIXED, name(intBits, fracBits, isSigned), intBits + fracBits),
    intBits(intBits),
    fracBits(fracBits),
    isSigned(isSigned) {}

  const std::size_t intBits;
  const std::size_t fracBits;
  const bool isSigned;

private:
  static std::string name(std::size_t intBits,
                          std::size_t fracBits,
                          bool isSigned) {
    return "fixed_" + std::to_string(intBits)  + "_"
                    + std::to_string(fracBits) + "_"
                    + std::to_string(isSigned);
  }
};

struct FloatType final: public Type {
  FloatType(std::size_t expBits, std::size_t precision):
    Type(FLOAT, name(expBits, precision), expBits + precision),
    expBits(expBits),
    precision(precision) {}

  const std::size_t expBits;
  const std::size_t precision;

private:
  static std::string name(std::size_t expBits, std::size_t precision) {
    return "float_" + std::to_string(expBits) + "_"
                    + std::to_string(precision);
  }
};

struct TupleType final: public Type {
  TupleType(const std::vector<const Type*> &types):
    Type(TUPLE, name(types), size(types)),
    types(types) {}

  const std::vector<const Type*> types;

private:
  static std::string name(const std::vector<const Type*> &types) {
    std::stringstream out;

    out << "tuple";
    for (const auto *type : types)
      out << "_" << type->name;

    return out.str();
  }

  static std::size_t size(const std::vector<const Type*> &types) {
    std::size_t sum = 0;
    for (const auto *type : types)
      sum += type->size;

    return sum;
  }
};

struct TensorType final: public Type {
  TensorType(const Type *type, const std::vector<std::size_t> &dims):
    Type(TENSOR, name(type, dims), size(type, dims)),
    type(type),
    dims(dims) {}

  const Type *type;
  const std::vector<std::size_t> dims;

private:
  static std::string name(const Type *type,
                          const std::vector<std::size_t> &dims) {
    std::stringstream out;

    out << "tensor_" << type->name;
    for (auto dim : dims)
      out << "_" << dim;

    return out.str();
  }

  static std::size_t size(const Type *type,
                          const std::vector<std::size_t> &dims) {
    std::size_t mul = 1;
    for (auto dim : dims)
      mul *= dim;

    return type->size * mul;
  }
};

//===----------------------------------------------------------------------===//
// Port
//===----------------------------------------------------------------------===//

struct Port final {
  Port(const std::string &name,
       const Type &type,
       float flow,
       unsigned latency,
       bool isConst,
       const long long &value):
    name(name),
    type(type),
    flow(flow),
    latency(latency),
    isConst(isConst),
    value(value) {}

  /// \deprecated
  Port(const std::string &name,
       const std::string &type,
       float flow,
       unsigned latency,
       bool isConst,
       const long long &value):
    Port(name, Type::get(type), flow, latency, isConst, value) {}
 
  const std::string name;
  const Type &type;
  const float flow;
  const unsigned latency;
  const bool isConst;
  const long long value;
};

//===----------------------------------------------------------------------===//
// NodeType
//===----------------------------------------------------------------------===//

struct NodeType final {
  NodeType(const std::string &name, Model &model):
    name(name), model(model) {}

  void addInput(Port *input) {
    inputs.push_back(input);
  }

  void addOutput(Port *output) {
    outputs.push_back(output);
  }

  Port* findInput(const std::string &name) const {
    auto i = std::find_if(inputs.begin(), inputs.end(),
      [&name](Port *port) { return port->name == name; });
    return i != inputs.end() ? *i : nullptr;
  }

  Port* findOutput(const std::string &name) const {
    auto i = std::find_if(outputs.begin(), outputs.end(),
      [&name](Port *port) { return port->name == name; });
    return i != outputs.end() ? *i : nullptr;
  }

  Signature getSignature() const {
    std::vector<std::string> inputTypeNames;
    std::vector<std::string> outputTypeNames;
    for (const auto *input : inputs) {
      inputTypeNames.push_back(input->type.name);
    }
    for (const auto *output : outputs) {
      outputTypeNames.push_back(output->type.name);
    }
    return Signature(name,
                     inputTypeNames,
                     outputTypeNames);
  }

  bool isConst() const {
    if (!inputs.empty())
      return false;

    for (const auto *output: outputs) {
      if (!output->isConst)
        return false;
    }

    return true;
  }

  bool isSource() const {
    return inputs.empty() && !isConst();
  }

  bool isMux() const {
    return outputs.size() == 1
        && starts_with(name, "MUX");
  }

  bool isSink() const {
    return outputs.empty();
  }

  bool isMerge() const {
    return outputs.size() == 1
        && starts_with(name, "merge");
  }

  bool isSplit() const {
    return inputs.size() == 1
        && starts_with(name, "split");
  }

  bool isDup() const {
    return inputs.size() == 1
        && starts_with(name, "dup");
  }

  bool isDelay() const {
    return inputs.size() == 1
        && outputs.size() == 1
        && starts_with(name, "delay");
  }
  
  const std::string name;
  std::vector<Port*> inputs;
  std::vector<Port*> outputs;

  /// Reference to the parent.
  Model &model;
};

//===----------------------------------------------------------------------===//
// Binding
//===----------------------------------------------------------------------===//

struct Binding final {
  Binding() = default;
  Binding(const Node *node, const Port *port):
    node(node), port(port) {}

  bool isLinked() const {
    return node != nullptr;
  }

  const Node *node = nullptr;
  const Port *port = nullptr;
};

//===----------------------------------------------------------------------===//
// Chan
//===----------------------------------------------------------------------===//

struct Chan final {
  Chan(const std::string &name, const std::string &type, Graph &graph, 
       const size_t latency = 0):
    name(name), type(type), graph(graph), latency(latency) {}

  std::string name;
  const std::string type;
  
  Binding source;
  Binding target;

  /// Reference to the parent.
  Graph &graph;

  size_t latency = 0;

  /// Indicators.
  ChanInd ind;
};

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

struct Node final {
  Node(const std::string &name, const NodeType &type, Graph &graph):
    name(name), type(type), graph(graph) {}

  void addInput(Chan *input) {
    inputs.push_back(input);
  }

  void addOutput(Chan *output) {
    outputs.push_back(output);
  }

  Chan* findInput(const std::string &name) const {
    auto i = std::find_if(inputs.begin(), inputs.end(),
      [&name](Chan *chan) { return chan->name == name; });
    return i != inputs.end() ? *i : nullptr;
  }

  Chan* findOutput(const std::string &name) const {
    auto i = std::find_if(outputs.begin(), outputs.end(),
      [&name](Chan *chan) { return chan->name == name; });
    return i != outputs.end() ? *i : nullptr;
  }

  bool isConst()  const { return type.isConst();  }
  bool isSource() const { return type.isSource(); }
  bool isSink()   const { return type.isSink();   }
  bool isMerge()  const { return type.isMerge();  }
  bool isSplit()  const { return type.isSplit();  }
  bool isDup()    const { return type.isDup();    }
  bool isDelay()  const { return type.isDelay();  }

  const std::string name;
  const NodeType &type;
  std::vector<Chan*> inputs;
  std::vector<Chan*> outputs;

  /// Reference to the parent.
  Graph &graph;
  /// Mapping.
  std::shared_ptr<library::MetaElement> map;
  /// Parameters.
  Parameters params;
  /// Indicators.
  NodeInd ind;
};

//===----------------------------------------------------------------------===//
// Graph
//===----------------------------------------------------------------------===//

struct Graph final {
  Graph(const std::string &name, Model &model):
    name(name), model(model) {}

  void addChan(Chan *chan) {
    chans.push_back(chan);
  }

  void addNode(Node *node) {
    nodes.push_back(node);

    if (node->inputs.empty()) {
      sources.push_back(node);
    } else if (node->outputs.empty()) {
      targets.push_back(node);
    }
  }

  Chan* findChan(const std::string &name) const {
    auto i = std::find_if(chans.begin(), chans.end(),
      [&name](Chan *chan) { return chan->name == name; });
    return i != chans.end() ? *i : nullptr;
  }

  Node* findNode(const std::string &name) const {
    auto i = std::find_if(nodes.begin(), nodes.end(),
      [&name](Node *node) { return node->name == name; });
    return i != nodes.end() ? *i : nullptr;
  }

  void instantiate(
    const Graph &graph,
    const std::string &name,
    const std::map<std::string, std::map<std::string, Chan*>> &inputs,
    const std::map<std::string, std::map<std::string, Chan*>> &outputs);

  //--------------------------------------------------------------------------//
  // Graph Interface
  //--------------------------------------------------------------------------//

  using V = Node*;
  using E = Chan*;

  std::size_t nNodes() const { return nodes.size(); }
  std::size_t nEdges() const { return chans.size(); }

  bool hasNode(Node *node) const { return true; }
  bool hasEdge(Chan *chan) const { return true; }

  const std::vector<Node*>& getSources() const {
    return sources;
  }

  const std::vector<Chan*>& getOutEdges(Node *node) const {
    return node->outputs;
  }

  Node* leadsTo(Chan *chan) const {
    return const_cast<Node*>(chan->target.node);
  }

  //--------------------------------------------------------------------------//
  // Fields
  //--------------------------------------------------------------------------//

  const std::string name;
  std::vector<Chan*> chans;
  std::vector<Node*> nodes;

  std::vector<Node*> sources;
  std::vector<Node*> targets;

  /// Reference to the parent.
  Model &model;
  /// Indicators.
  GraphInd ind;
};

//===----------------------------------------------------------------------===//
// Model
//===----------------------------------------------------------------------===//

struct Model final {
  Model(const std::string &name):
    name(name) {}

  void addNodetype(const Signature &signature, NodeType *nodetype) {
    /*std::cout << signature.name << std::endl;
    std::cout << "Inputs:" << std::endl;
    std::cout << "***********************************************" << std::endl;
    for (const auto &inputTypeName : signature.inputTypeNames) {
      std::cout << inputTypeName << std::endl;
    }
    std::cout << "***********************************************" << std::endl;
    std::cout << "Outputs:" << std::endl;
    std::cout << "***********************************************" << std::endl;
    for (const auto &outputTypeName : signature.outputTypeNames) {
      std::cout << outputTypeName << std::endl;
    }
    std::cout << "*********************************************" << std::endl;*/
    nodetypes.insert({ signature, nodetype });
  }

  void addGraph(Graph *graph) {
    graphs.push_back(graph);
  }
  
  NodeType* findNodetype(const Signature &signature) const {
    /*std::cout << signature.name << std::endl;
    std::cout << "Inputs:" << std::endl;
    std::cout << "***********************************************" << std::endl;
    for (const auto &inputTypeName : signature.inputTypeNames) {
      std::cout << inputTypeName << std::endl;
    }
    std::cout << "***********************************************" << std::endl;
    std::cout << "Outputs:" << std::endl;
    std::cout << "***********************************************" << std::endl;
    for (const auto &outputTypeName : signature.outputTypeNames) {
      std::cout << outputTypeName << std::endl;
    }
    std::cout << "*********************************************" << std::endl;*/
    auto nodeTypeIterator = nodetypes.find(signature);
    return nodeTypeIterator != nodetypes.end() ? nodeTypeIterator->second :
                                                 nullptr;
    // deprecated
    /*auto i = std::find_if(nodetypes.begin(), nodetypes.end(),
      [&name](NodeType *nodetype) { return nodetype->name == name; });
    return i != nodetypes.end() ? *i : nullptr;*/
  }

  Graph* findGraph(const std::string &name) const {
    auto i = std::find_if(graphs.begin(), graphs.end(),
      [&name](Graph *graph) { return graph->name == name; });
    return i != graphs.end() ? *i : nullptr;
  }

  Graph* main() const {
    return graphs.size() == 1 ? *graphs.begin() : findGraph("main");
  }

  void save();
  void undo();

  void insertDelay(Chan &chan, unsigned latency);

  const std::string name;
  // deprecated
  //std::vector<NodeType*> nodetypes;
  std::unordered_map<Signature, NodeType*> nodetypes;
  std::vector<Graph*> graphs;

  std::vector<Transform*> transforms;

  /// Indicators.
  ModelInd ind;
};

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

std::ostream& operator<<(std::ostream &out, const Type &type);
std::ostream& operator<<(std::ostream &out, const Port &port);
std::ostream& operator<<(std::ostream &out, const NodeType &nodetype);
std::ostream& operator<<(std::ostream &out, const Chan &chan);
std::ostream& operator<<(std::ostream &out, const Node &node);
std::ostream& operator<<(std::ostream &out, const Graph &graph);
std::ostream& operator<<(std::ostream &out, const Model &model);

} // namespace eda::hls::model
