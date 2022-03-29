//===- Model.h - MLIR model ------------------*- C++ -*--------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"

#include <string>

#include "HIL/Ops.h"
#include "hls/model/model.h"

namespace mlir::model {
class MLIRModule {
public:
  MLIRModule(MLIRModule &&oth);
  MLIRModule &operator=(MLIRModule &&oth);
  MLIRModule clone();
  MLIRContext *get_context();
  static MLIRModule load_from_mlir(const std::string &s);
  static MLIRModule load_from_mlir_file(const std::string &filename);
  static MLIRModule load_from_model(const eda::hls::model::Model &m);
  void print(llvm::raw_ostream &os);
  mlir::hil::Model get_root();

private:
  MLIRModule(std::shared_ptr<mlir::MLIRContext> context,
             mlir::OwningOpRef<mlir::ModuleOp> &&module);
  std::shared_ptr<mlir::MLIRContext> context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};
} // namespace mlir::model

namespace eda::hls::model {

std::shared_ptr<eda::hls::model::Model>
parse_model_from_mlir(const std::string &s);
std::shared_ptr<eda::hls::model::Model>
parse_model_from_mlir_file(const std::string &filename);
} // namespace eda::hls::model
