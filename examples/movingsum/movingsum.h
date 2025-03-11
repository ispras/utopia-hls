//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

class MovingSum : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "MovingSum";
  }

  ~MovingSum() override = default;

  MovingSum() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType &type = dfUInt(32);
    DFVariable x = io.input("x", type);
    DFVariable xMinus = offset(x, -1);
    DFVariable xPlus = offset(x, 1);
    DFVariable sum1 = xMinus + x;
    DFVariable sum2 = sum1 + xPlus;
    DFVariable out = io.output("out", type);
    out.connect(sum2);
  }
};
