//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_IR_BUILDER_H
#define DFCXX_IR_BUILDER_H

#include "dfcir/DFCIROperations.h"
#include "dfcxx/IRbuilders/converter.h"
#include "dfcxx/kernel.h"
#include "dfcxx/typedefs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace dfcxx {

class DFCIRBuilder {
private:
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  DFCIRTypeConverter conv;

  void translate(Node node, const Graph &graph, mlir::OpBuilder &builder,
                 std::unordered_map<Node, mlir::Value> &map);
  
  void buildKernelBody(const Graph &graph, mlir::OpBuilder &builder);

  mlir::dfcir::KernelOp buildKernel(Kernel *kern, mlir::OpBuilder &builder);

public:
  DFCIRBuilder();

  mlir::ModuleOp buildModule(Kernel *kern);
};

} // namespace dfcxx

#endif // DFCXX_IR_BUILDER_H
