//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
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
  void print(llvm::raw_fd_ostream &os);
  mlir::hil::Model get_root();

private:
  MLIRModule(std::shared_ptr<mlir::MLIRContext> context,
             mlir::OwningModuleRef &&module);
  std::shared_ptr<mlir::MLIRContext> context_;
  mlir::OwningModuleRef module_;
};
} // namespace mlir::model

namespace eda::hls::model {

std::shared_ptr<eda::hls::model::Model>
parse_model_from_mlir(const std::string &s);
std::shared_ptr<eda::hls::model::Model>
parse_model_from_mlir_file(const std::string &filename);
} // namespace eda::hls::model
