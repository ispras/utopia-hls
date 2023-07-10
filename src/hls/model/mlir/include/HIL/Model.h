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

namespace mlir::model {

class MLIRModule {
public:
  MLIRModule(MLIRModule &&oth);
  MLIRModule &operator=(MLIRModule &&oth);
  MLIRModule clone();
  MLIRContext *getContext();
  static MLIRModule loadFromMlir(const std::string &string);
  static MLIRModule loadFromMlirFile(const std::string &filename);
  static MLIRModule loadFromModel(const eda::hls::model::Model &model);
  void print(llvm::raw_ostream &os);
  mlir::hil::Model getRoot();

private:
  MLIRModule(std::shared_ptr<mlir::MLIRContext> context,
             mlir::OwningOpRef<mlir::ModuleOp> &&module);
  std::shared_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

} // namespace mlir::model

namespace eda::hls::model {

std::shared_ptr<eda::hls::model::Model>
parseModelFromMlir(const std::string &string);
std::shared_ptr<eda::hls::model::Model>
parseModelFromMlirFile(const std::string &filename);

} // namespace eda::hls::model