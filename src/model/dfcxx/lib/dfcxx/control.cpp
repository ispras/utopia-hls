//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/control.h"
#include "dfcxx/vars/vars.h"

using IODirection = dfcxx::DFVariableImpl::IODirection;

namespace dfcxx {

Control::Control(KernelMeta &meta) : meta(meta) {}

DFVariable Control::mux(DFVariable ctrl,
                        std::initializer_list<DFVariable> args) {
  const DFVariable *argsData = args.begin();
  unsigned argsCount = args.size();
  auto *var = meta.varBuilder.buildStream("",
                                          IODirection::NONE,
                                          &meta,
                                          argsData[0].getType());
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::MUX, NodeData {.muxId = 0});
  meta.graph.addChannel(ctrl, var, 0, false);
  for (unsigned i = 0; i < argsCount; ++i) {
    meta.graph.addChannel(argsData[i], var, i + 1, false);
  }
  return var;
}

} // namespace dfcxx
