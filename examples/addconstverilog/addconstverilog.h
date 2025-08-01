//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

class AddConstVerilog : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "AddConstVerilog";
  }

  ~AddConstVerilog() override = default;

  AddConstVerilog() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType type = dfUInt(32);
    DFVariable x = io.input("x", type);
    DFVariable out = io.output("out", type);
    instanceExt("Adder", {{x, "in"}}, {{out, type, "out"}}, {});
  }
};
