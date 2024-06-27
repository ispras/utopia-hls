//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/control.h"
#include "dfcxx/vars/vars.h"

namespace dfcxx {

Control::Control(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &varBuilder,
                 KernStorage &storage) : graph(graph),
                                         helper(graph, typeBuilder,
                                                varBuilder, storage),
                                         varBuilder(varBuilder),
                                         storage(storage) {}

DFVariable Control::mux(DFVariable ctrl,
                        std::initializer_list<DFVariable> args) {
  const DFVariable *argsData = args.begin();
  unsigned argsCount = args.size();
  DFVariable var = helper.varBuilder.buildMuxCopy(argsData[0], helper);
  storage.addVariable(var);
  graph.addNode(var, OpType::MUX, NodeData{.muxId = 0});
  graph.addChannel(ctrl, var, 0, false);
  for (unsigned i = 0; i < argsCount; ++i) {
    graph.addChannel(argsData[i], var, i + 1, false);
  }
  return var;
}

} // namespace dfcxx
