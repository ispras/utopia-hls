//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

class Scalar3 : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "Scalar3";
  }

  ~Scalar3() override = default;

  Scalar3() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType type = dfUInt(32);
    DFVariable x1 = io.input("x1", type);
    DFVariable y1 = io.input("y1", type);
    DFVariable x2 = io.input("x2", type);
    DFVariable y2 = io.input("y2", type);
    DFVariable x3 = io.input("x3", type);
    DFVariable y3 = io.input("y3", type);

    DFVariable mul1 = x1 * y1;
    DFVariable mul2 = x2 * y2;
    DFVariable mul3 = x3 * y3;

    DFVariable sum1 = mul1 + mul2;
    DFVariable sum2 = sum1 + mul3;

    DFVariable out = io.output("out", type);
    out.connect(sum2);
  }
};
