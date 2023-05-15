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
// MLIR transformer API.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "HIL/Combine.h"
#include "HIL/Model.h"
#include "hls/model/model.h"

#include <iostream>

using eda::hls::model::Model;
using mlir::model::MLIRModule;

namespace mlir::transforms {

template <typename T> class Transformer {
public:
  Transformer(MLIRModule &&module);
  Transformer(const Model &model);
  Transformer(Transformer &&oth);
  void apply_transform(std::function<void(MLIRModule &)> transform);
  void undo_transforms();
  T done();

private:
  MLIRModule module_;
  MLIRModule module_init_;
};

/* Copy constructors */
template <>
inline Transformer<MLIRModule>::Transformer(MLIRModule &&module)
    : module_(std::move(module)), module_init_(module_.clone()) {}
template <>
inline Transformer<Model>::Transformer(const Model &model)
    : module_(MLIRModule::load_from_model(model)),
      module_init_(module_.clone()) {}
template <typename T>
Transformer<T>::Transformer(Transformer &&oth)
    : module_(std::move(oth.module_)),
      module_init_(std::move(oth.module_init_)) {}

/* Transform-related methods */
template <typename T>
void Transformer<T>::apply_transform(
    std::function<void(MLIRModule &)> transform) {
  transform(module_);
}

template <typename T> void Transformer<T>::undo_transforms() {
  module_ = module_init_.clone();
}

/* End-of-transformation methods */
template <> inline MLIRModule Transformer<MLIRModule>::done() {
  (void)std::move(module_init_);
  return std::move(module_);
}

template <> inline Model Transformer<Model>::done() {
  (void)std::move(module_init_);
  std::string buf;
  llvm::raw_string_ostream os{buf};
  module_.print(os);
  auto model = std::move(*eda::hls::model::parse_model_from_mlir(buf));
  return model;
}

/* Transformations */
std::function<void(MLIRModule &)> ChanAddSourceTarget();
std::function<void(MLIRModule &)> InsertDelay(std::string chan_name,
                                              unsigned latency);
std::function<void(MLIRModule &)> UnfoldInstance(std::string instance_name);

} // namespace mlir::transforms