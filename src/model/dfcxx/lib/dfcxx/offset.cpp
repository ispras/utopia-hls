//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/offset.h"
#include "dfcxx/vars/vars.h"

namespace dfcxx {

Offset::Offset(KernMeta &meta) : meta(meta) {}

DFVariable Offset::operator()(DFVariable &stream, int64_t offset) {
  if (!stream.isStream()) { throw std::exception(); }
  auto *var = meta.varBuilder.buildStream("",
                                          DFVariableImpl::IODirection::NONE,
                                          &meta, stream.getType());
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::OFFSET, NodeData{.offset = offset});
  meta.graph.addChannel(stream, var, 0, false);
  return var;
}

} // namespace dfcxx
