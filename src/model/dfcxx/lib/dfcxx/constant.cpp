//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/constant.h"

namespace dfcxx {

Constant::Constant(Graph &graph, TypeBuilder &typeBuilder,
                   VarBuilder &varBuilder, KernStorage &storage) : 
                   graph(graph),
                   helper(graph, typeBuilder, varBuilder, storage),
                   varBuilder(varBuilder), storage(storage) {}

DFVariable Constant::var(const DFType &type, int64_t value) {
  DFVariable var = varBuilder.buildConstant(helper, type,
                                            ConstantTypeKind::INT,
                                            ConstantValue{.int_ = value});
  storage.addVariable(var);
  graph.addNode(var, OpType::CONST, NodeData{});
  return var;
}

DFVariable Constant::var(const DFType &type, uint64_t value) {
  DFVariable var = varBuilder.buildConstant(helper, *(type.getImpl()),
                                            ConstantTypeKind::UINT,
                                            ConstantValue{.uint_ = value});
  storage.addVariable(var);
  graph.addNode(var, OpType::CONST, NodeData{});
  return var;
}

DFVariable Constant::var(const DFType &type, double value) {
  DFVariable var = varBuilder.buildConstant(helper, *(type.getImpl()),
                                            ConstantTypeKind::FLOAT,
                                            ConstantValue{.double_ = value});
  storage.addVariable(var);
  graph.addNode(var, OpType::CONST, NodeData{});
  return var;
}

} // namespace dfcxx
