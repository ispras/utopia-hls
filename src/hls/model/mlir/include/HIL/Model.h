//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//
// MLIR model.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "HIL/Ops.h"
#include "hls/model/model.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"

#include <string>

using HILDialect = mlir::hil::HILDialect;
using MLIRContext = mlir::MLIRContext;
using Model = eda::hls::model::Model;
using ModuleOp = mlir::ModuleOp;
using BindingGraphAttr = mlir::hil::BindingGraphAttr;
using BindingAttr = mlir::hil::BindingAttr;
using ChansOp = mlir::hil::ChansOp;
using ChanOp = mlir::hil::ChanOp;
using ConsOp = mlir::hil::ConsOp;
using ConOp = mlir::hil::ConOp;
using GraphsOp = mlir::hil::GraphsOp;
using GraphOp = mlir::hil::GraphOp;
using ModelOp = mlir::hil::ModelOp;
using NodeTypesOp = mlir::hil::NodeTypesOp;
using NodeTypeOp = mlir::hil::NodeTypeOp;
using NodesOp = mlir::hil::NodesOp;
using NodeOp = mlir::hil::NodeOp;
template<typename Type>
using OwningOpRef = mlir::OwningOpRef<Type>;
using PortAttr = mlir::hil::PortAttr;

namespace mlir::model {

class MLIRModule {
public:
  MLIRModule(MLIRModule &&oth);
  MLIRModule &operator=(MLIRModule &&oth);
  MLIRModule clone();
  MLIRContext *getContext();
  static MLIRModule loadFromMlir(const std::string &string);
  static MLIRModule loadFromMlirFile(const std::string &filename);
  static MLIRModule loadFromModel(const Model &model);
  void print(llvm::raw_ostream &os);
  ModelOp getRoot();

private:
  MLIRModule(std::shared_ptr<MLIRContext> context,
             mlir::OwningOpRef<ModuleOp> &&module);
  std::shared_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
};

} // namespace mlir::model

namespace eda::hls::model {
std::shared_ptr<Model>
parseModelFromMlir(const std::string &string);
std::shared_ptr<Model>
parseModelFromMlirFile(const std::string &filename);

} // namespace eda::hls::model