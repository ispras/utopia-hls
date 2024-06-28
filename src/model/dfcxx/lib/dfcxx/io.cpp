//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/io.h"

namespace dfcxx {

using IODirection = dfcxx::IODirection;

IO::IO(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &varBuilder,
       KernStorage &storage) : graph(graph), helper(graph,
                                                    typeBuilder,
                                                    varBuilder,
                                                    storage),
                               varBuilder(varBuilder),
                               storage(storage) {}

DFVariable IO::input(const std::string &name, const DFType &type) {
  DFVariable var = varBuilder.buildStream(name, 
                                          IODirection::INPUT,
                                          helper,
                                          type);
  storage.addVariable(var);
  graph.addNode(var, OpType::IN, NodeData{});
  return var;
}

DFVariable IO::inputScalar(const std::string &name, const DFType &type) {
  DFVariable var = varBuilder.buildScalar(name,
                                          IODirection::INPUT,
                                          helper,
                                          type);
  storage.addVariable(var);
  graph.addNode(var, OpType::IN, NodeData{});
  return var;
}

DFVariable IO::output(const std::string &name, const DFType &type) {
  DFVariable var = varBuilder.buildStream(name,
                                          IODirection::OUTPUT,
                                          helper,
                                          type);
  storage.addVariable(var);
  graph.addNode(var, OpType::OUT, NodeData{});
  return var;
}

DFVariable IO::outputScalar(const std::string &name, const DFType &type) {
  DFVariable var = varBuilder.buildScalar(name,
                                          IODirection::OUTPUT,
                                          helper,
                                          type);
  storage.addVariable(var);
  graph.addNode(var, OpType::OUT, NodeData{});
  return var;
}

} // namespace dfcxx
