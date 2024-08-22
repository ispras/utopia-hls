//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/constant.h"

namespace dfcxx {

Constant::Constant(KernMeta &meta) : meta(meta) {} 

DFVariable Constant::var(const DFType &type, int64_t value) {
  auto *var = meta.varBuilder.buildConstant(meta, type,
                                            DFConstant::Value{
                                              .int_ = value
                                            });
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::CONST, NodeData{});
  return var;
}

DFVariable Constant::var(const DFType &type, uint64_t value) {
  auto *var = meta.varBuilder.buildConstant(meta, type,
                                            DFConstant::Value{
                                              .uint_ = value
                                            });
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::CONST, NodeData{});
  return var;
}

DFVariable Constant::var(const DFType &type, double value) {
  auto *var = meta.varBuilder.buildConstant(meta, type,
                                            DFConstant::Value{
                                              .double_ = value
                                            });
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::CONST, NodeData{});
  return var;
}

} // namespace dfcxx
