//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//

#include "HIL/API.h"

namespace mlir::transforms {
using mlir::model::MLIRModule;

Transformer::Transformer(MLIRModule &&module)
    : module_(std::move(module)), module_init_(module_.clone()) {}
Transformer::Transformer(Transformer &&oth)
    : module_(std::move(oth.module_)),
      module_init_(std::move(oth.module_init_)) {}
void Transformer::apply_transform(std::function<void(MLIRModule &)> transform) {
  transform(module_);
}
void Transformer::undo_transforms() { module_ = module_init_.clone(); }

MLIRModule Transformer::done() {
  (void)std::move(module_init_);
  return std::move(module_);
}
} // namespace mlir::transforms
