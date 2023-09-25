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
#include "HIL/Conversion.h"
#include "HIL/Model.h"
#include "hls/model/model.h"
#include "mlir/Pass/PassManager.h"

#include <iostream>

using Model = eda::hls::model::Model;
using MLIRModule = mlir::model::MLIRModule;
using PassManager = mlir::PassManager;
using OpPassManager = mlir::OpPassManager;

namespace mlir::transforms {

template <typename T> class Transformer {
public:
  Transformer(MLIRModule &&module);
  Transformer(const Model &model);
  Transformer(Transformer &&oth);
  void applyTransform(std::function<void(MLIRModule &)> transform);
  void undoTransforms();
  void addPass(std::unique_ptr<Pass> pass);
  void runPasses();
  void clearPasses();
  void print();
  T done();

private:
  MLIRModule module;
  MLIRModule moduleInitial;
  PassManager passManager;
};

/* Copy constructors. */
template <>
inline Transformer<MLIRModule>::Transformer(MLIRModule &&module)
    : module(std::move(module)),
      moduleInitial(module.clone()),
      passManager(module.getContext()) {}
template <>
inline Transformer<Model>::Transformer(const Model &model)
    : module(MLIRModule::loadFromModel(model)),
      moduleInitial(module.clone()),
      passManager(module.getContext()) {}
template <typename T>
Transformer<T>::Transformer(Transformer &&oth)
    : module(std::move(oth.module)),
      moduleInitial(std::move(oth.moduleInitial)),
      passManager(std::move(oth.passManager)) {}

/* Transform-related methods */
template <typename T>
void Transformer<T>::applyTransform(
    std::function<void(MLIRModule &)> transform) {
  transform(module);
}

template <typename T>
void Transformer<T>::addPass(std::unique_ptr<Pass> pass) {
  passManager.addPass(std::move(pass));
}

template <typename T>
void Transformer<T>::runPasses() {
  ModuleOp moduleOp = mlir::cast<ModuleOp>(module.getRoot()->getParentOp());
  if (failed(passManager.run(moduleOp))) {
    std::cout << "Some passes failed!\n" << std::endl;
  }
}

template <typename T>
void Transformer<T>::clearPasses() {
  passManager.clear();
}

template <typename T> void Transformer<T>::undoTransforms() {
  module = moduleInitial.clone();
}

template <typename T> void Transformer<T>::print() {
  std::string buf;
  llvm::raw_string_ostream os{buf};
  module.print(os);
  std::cout << buf << std::endl;
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