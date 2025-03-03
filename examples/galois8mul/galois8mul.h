//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

class Galois8Mul : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "Galois8Mul";
  }

  ~Galois8Mul() override = default;

  using DFType = dfcxx::DFType;
  using DFVariable = dfcxx::DFVariable;

  Galois8Mul() : dfcxx::Kernel() {
    const DFType ioType = dfUInt(8);

    DFVariable left = io.input("left", ioType);
    DFVariable right = io.input("right", ioType);
    
    DFVariable c0 = constant.var(ioType, uint64_t(0));
    DFVariable c195 = constant.var(ioType, uint64_t(195));
    DFVariable currValue = c0;

    for (int i = 0; i < 7; ++i) {
      DFVariable isBitSet = right(0, 0);
      currValue = currValue ^ control.mux(isBitSet, {c0, left});
      DFVariable aboutToOverflow = left(7, 7);
      DFVariable muxed = control.mux(aboutToOverflow, {c0, c195});
      left = (left << 1) ^ muxed;
      right = right >> 1;
    }

    DFVariable isBitSet = right(0, 0);
    currValue = currValue ^ control.mux(isBitSet, {c0, left});

    DFVariable result = io.output("result", ioType);
    result.connect(currValue);
  }
};
