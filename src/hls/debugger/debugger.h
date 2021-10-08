//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

#include "z3++.h"

#include <memory>

using namespace eda::hls::model;

namespace eda::hls::debugger {

class Verifier final {

public:
  static Verifier& get() {
    if (instance == nullptr) {
      instance = std::unique_ptr<Verifier>(new Verifier());
    }
    return *instance;
  }

  bool equivalent(Model &left, Model &right) const;

private:
  Verifier() {}

  static std::unique_ptr<Verifier> instance;

  z3::expr to_expr(Model &model) const;
};
} // namespace eda::hls::debugger
