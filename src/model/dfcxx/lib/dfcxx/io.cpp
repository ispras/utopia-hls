//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/io.h"

namespace dfcxx {

using IODirection = dfcxx::DFVariableImpl::IODirection;

IO::IO(KernMeta &meta) : meta(meta) {}

DFVariable IO::input(const std::string &name, const DFType &type) {
  auto *var = meta.varBuilder.buildStream(name,
                                          IODirection::INPUT,
                                          &meta,
                                          type);
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::IN, NodeData {});
  return var;
}

DFVariable IO::inputScalar(const std::string &name, const DFType &type) {
  auto *var = meta.varBuilder.buildScalar(name,
                                          IODirection::INPUT,
                                          &meta,
                                          type);
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::IN, NodeData {});
  return var;
}

DFVariable IO::newStream(const DFType &type) {
  auto *var = meta.varBuilder.buildStream("",
                                          IODirection::NONE,
                                          &meta,
                                          type);
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::NONE, NodeData {});
  return var;
}

DFVariable IO::newScalar(const DFType &type) {
  auto *var = meta.varBuilder.buildScalar("",
                                          IODirection::NONE,
                                          &meta,
                                          type);
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::NONE, NodeData {});
  return var;
}

DFVariable IO::output(const std::string &name, const DFType &type) {
  auto *var = meta.varBuilder.buildStream(name,
                                          IODirection::OUTPUT,
                                          &meta,
                                          type);
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::OUT, NodeData {});
  return var;
}

DFVariable IO::outputScalar(const std::string &name, const DFType &type) {
  auto *var = meta.varBuilder.buildScalar(name,
                                          IODirection::OUTPUT,
                                          &meta,
                                          type);
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::OUT, NodeData {});
  return var;
}

} // namespace dfcxx
