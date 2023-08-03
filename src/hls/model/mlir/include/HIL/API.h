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

using Model = eda::hls::model::Model;
using MLIRModule = mlir::model::MLIRModule;

namespace mlir::transforms {

template <typename T> class Transformer {
public:
  Transformer(MLIRModule &&module);
  Transformer(const Model &model);
  Transformer(Transformer &&oth);
  void applyTransform(std::function<void(MLIRModule &)> transform);
  void undoTransforms();
  T done();

private:
  MLIRModule module;
  MLIRModule moduleInitial;
};

/* Copy constructors. */
template <>
inline Transformer<MLIRModule>::Transformer(MLIRModule &&module)
    : module(std::move(module)), moduleInitial(module.clone()) {}
template <>
inline Transformer<Model>::Transformer(const Model &model)
    : module(MLIRModule::loadFromModel(model)),
      moduleInitial(module.clone()) {}
template <typename T>
Transformer<T>::Transformer(Transformer &&oth)
    : module(std::move(oth.module)),
      moduleInitial(std::move(oth.moduleInitial)) {}

/* Transform-related methods */
template <typename T>
void Transformer<T>::applyTransform(
    std::function<void(MLIRModule &)> transform) {
  transform(module);
}

template <typename T> void Transformer<T>::undoTransforms() {
  module = moduleInitial.clone();
}

/* End-of-transformation methods */
template <> inline MLIRModule Transformer<MLIRModule>::done() {
  (void)std::move(moduleInitial);
  return std::move(module);
}

template <> inline Model Transformer<Model>::done() {
  (void)std::move(moduleInitial);
  std::string buf;
  llvm::raw_string_ostream os{buf};
  module.print(os);
  std::cout << buf << std::endl;
  auto model = std::move(*eda::hls::model::parseModelFromMlir(buf));
  return model;
}

/* Transformations */
std::function<void(MLIRModule &)> ChanAddSourceTarget();
std::function<void(MLIRModule &)> InsertDelay(const std::string &chanName,
                                              const unsigned latency);
std::function<void(MLIRModule &)> UnfoldInstance(
    const std::string &instanceName, const std::string &instanceGraphName,
    const std::string &mainGraphName);

} // namespace mlir::transforms