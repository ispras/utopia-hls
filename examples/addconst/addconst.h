//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

class AddConst : public dfcxx::Kernel {
public:
  std::string_view getName() override {
    return "AddConst";
  }

  ~AddConst() override = default;

  AddConst() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType type = dfUInt(32);
    DFVariable x = io.input("x", type);
    DFVariable const5 = constant.var(type, uint64_t(5));
    DFVariable sum1 = x + const5;
    DFVariable sum2 = sum1 + x;
    DFVariable out = io.output("out", type);
    out.connect(sum2);
  }
};
