//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

class Polynomial2 : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "Polynomial2";
  }

  ~Polynomial2() override = default;

  Polynomial2() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType &type = dfUInt(32);
    DFVariable x = io.input("x", type);
    DFVariable squared = x * x;
    DFVariable squaredPlusX = squared + x;
    DFVariable result = squaredPlusX + x;
    DFVariable out = io.output("out", type);
    out.connect(result);
  }
};
