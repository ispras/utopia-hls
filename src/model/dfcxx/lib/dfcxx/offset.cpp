//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/offset.h"
#include "dfcxx/vars/vars.h"

namespace dfcxx {

Offset::Offset(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &varBuilder,
               KernStorage &storage) : graph(graph), helper(graph, typeBuilder,
                                                            varBuilder, storage),
               varBuilder(varBuilder), storage(storage) {}

DFVariable Offset::operator()(DFVariable &stream, int64_t offset) {
  if (!stream.isStream()) { throw std::exception(); }
  DFVariable var = varBuilder.buildStream("", IODirection::NONE, helper,
                                          stream.getType());
  storage.addVariable(var);
  graph.addNode(var, OpType::OFFSET, NodeData{.offset = offset});
  graph.addChannel(stream, var, 0, false);
  return var;
}

} // namespace dfcxx