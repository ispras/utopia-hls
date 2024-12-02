//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

class MuxMul : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "MuxMul";
  }

  ~MuxMul() override = default;

  MuxMul() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType type = dfUInt(32);
    const DFType ctrl_type = dfUInt(1);
    DFVariable x = io.input("x", type);
    DFVariable ctrl = io.input("ctrl", ctrl_type);
    DFVariable c1 = constant.var(type, uint64_t(0));
    DFVariable c2 = constant.var(type, uint64_t(1));
    DFVariable muxed = control.mux(ctrl, {c1, c2});
    DFVariable sum = x + x;
    DFVariable product = sum * muxed;
    DFVariable out = io.output("out", type);
    out.connect(product);
  }
};
