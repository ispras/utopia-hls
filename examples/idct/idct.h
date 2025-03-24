//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

#include <cassert>

// Input matrix element width.
static constexpr uint32_t WIN = 12;
// Width used in the computations.
static constexpr uint32_t WBUF = 32;
// Width of communication channel between calculators for rows and columns.
static constexpr uint32_t WIM = 13;
// Output matrix element width.
static constexpr uint32_t WOUT = 9;
// Matrix dimension size.
static constexpr uint32_t DIM = 8;
// Matrix elements count.
static constexpr uint32_t SIZE = DIM * DIM;

class IDCT : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "IDCT";
  }

  ~IDCT() override = default;

  using DFType = dfcxx::DFType;
  using DFVariable = dfcxx::DFVariable;

  DFVariable iclp(DFVariable var, DFVariable left, DFVariable right) {
    const DFType muxType = dfUInt(2);
    DFVariable firstRange = (var >= left).cast(muxType);
    DFVariable secondRange = (var > right).cast(muxType);
    DFVariable muxValue = firstRange + secondRange;
    return control.mux(muxValue, {left, var, right});
  }

  IDCT() : dfcxx::Kernel() {
    const DFType inType = dfInt(WIN * SIZE);
    const DFType bufType = dfInt(WBUF);
    const DFType intermType = dfInt(WIM);
    const DFType outElementType = dfInt(WOUT);
    const DFType outType = dfInt(WOUT * SIZE);

    DFVariable const128 = constant.var(bufType, int64_t(128));
    DFVariable const181 = constant.var(bufType, int64_t(181));
    DFVariable const8192 = constant.var(bufType, int64_t(8192));
    DFVariable const4 = constant.var(bufType, int64_t(4));
    
    DFVariable constM256 = constant.var(intermType, int64_t(-256));
    DFVariable const255 = constant.var(intermType, int64_t(255));
    
    DFVariable W1 = constant.var(bufType, int64_t(2841));
    DFVariable W2 = constant.var(bufType, int64_t(2676));
    DFVariable W3 = constant.var(bufType, int64_t(2408));
    DFVariable W5 = constant.var(bufType, int64_t(1609));
    DFVariable W6 = constant.var(bufType, int64_t(1108));
    DFVariable W7 = constant.var(bufType, int64_t(565));
    
    std::vector<DFVariable> values;

    DFVariable x = io.input("x", inType);
    
    for (unsigned i = 0; i < SIZE; ++i) {
      uint32_t rightInd = i * WIN;
      uint32_t leftInd = (WIN - 1) + rightInd;
      values.push_back(x(leftInd, rightInd));
    }

    for (unsigned i = 0; i < DIM; ++i) {
      DFVariable x0 = ((values[DIM * i + 0]).cast(bufType) << 11) + const128;
      DFVariable x1 = ((values[DIM * i + 4]).cast(bufType) << 11);
      DFVariable x2 = values[DIM * i + 6].cast(bufType);
      DFVariable x3 = values[DIM * i + 2].cast(bufType);
      DFVariable x4 = values[DIM * i + 1].cast(bufType);
      DFVariable x5 = values[DIM * i + 7].cast(bufType);
      DFVariable x6 = values[DIM * i + 5].cast(bufType);
      DFVariable x7 = values[DIM * i + 3].cast(bufType);
      
      DFVariable x8 = (x4 + x5) * W7;
      x4 = x8 + x4 * (W1 - W7);
      x5 = x8 - x5 * (W1 + W7);
      x8 = (x6 + x7) * W3;
      x6 = x8 - x6 * (W3 - W5);
      x7 = x8 - x7 * (W3 + W5);
      
      x8 = x0 + x1;
      x0 = x0 - x1;
      x1 = (x3 + x2) * W6;
      x2 = x1 - x2 * (W2 + W6);
      x3 = x1 + x3 * (W2 - W6);
      x1 = x4 + x6;
      x4 = x4 - x6;
      x6 = x5 + x7;
      x5 = x5 - x7;
      
      x7 = x8 + x3;
      x8 = x8 - x3;
      x3 = x0 + x2;
      x0 = x0 - x2;
      x2 = (((x4 + x5) * const181) + const128) >> 8;
      x4 = (((x4 - x5) * const181) + const128) >> 8;
      
      values[DIM * i + 0] = ((x7 + x1) >> 8).cast(intermType);
      values[DIM * i + 1] = ((x3 + x2) >> 8).cast(intermType);
      values[DIM * i + 2] = ((x0 + x4) >> 8).cast(intermType);
      values[DIM * i + 3] = ((x8 + x6) >> 8).cast(intermType);
      values[DIM * i + 4] = ((x8 - x6) >> 8).cast(intermType);
      values[DIM * i + 5] = ((x0 - x4) >> 8).cast(intermType);
      values[DIM * i + 6] = ((x3 - x2) >> 8).cast(intermType);
      values[DIM * i + 7] = ((x7 - x1) >> 8).cast(intermType);
    }
    
    for (unsigned i = 0; i < DIM; ++i) {
      DFVariable x0 = (values[DIM * 0 + i].cast(bufType) << 8) + const8192;
      DFVariable x1 = (values[DIM * 4 + i].cast(bufType) << 8);
      DFVariable x2 = values[DIM * 6 + i].cast(bufType);
      DFVariable x3 = values[DIM * 2 + i].cast(bufType);
      DFVariable x4 = values[DIM * 1 + i].cast(bufType);
      DFVariable x5 = values[DIM * 7 + i].cast(bufType);
      DFVariable x6 = values[DIM * 5 + i].cast(bufType);
      DFVariable x7 = values[DIM * 3 + i].cast(bufType);
      
      DFVariable x8 = ((x4 + x5) * W7) + const4;
      x4 = (x8 + x4 * (W1 - W7)) >> 3;
      x5 = (x8 - x5 * (W1 + W7)) >> 3;
      x8 = ((x6 + x7) * W3) + const4;
      x6 = (x8 - x6 * (W3 - W5)) >> 3;
      x7 = (x8 - x7 * (W3 + W5)) >> 3;
      
      x8 = x0 + x1;
      x0 = x0 - x1;
      x1 = ((x3 + x2) * W6) + const4;
      x2 = (x1 - x2 * (W2 + W6)) >> 3;
      x3 = (x1 + x3 * (W2 - W6)) >> 3;
      x1 = x4 + x6;
      x4 = x4 - x6;
      x6 = x5 + x7;
      x5 = x5 - x7;
      
      x7 = x8 + x3;
      x8 = x8 - x3;
      x3 = x0 + x2;
      x0 = x0 - x2;
      x2 = (((x4 + x5) * const181) + const128) >> 8;
      x4 = (((x4 - x5) * const181) + const128) >> 8;

      values[DIM * 0 + i] = iclp(((x7 + x1) >> 14).cast(intermType), constM256, const255);
      values[DIM * 1 + i] = iclp(((x3 + x2) >> 14).cast(intermType), constM256, const255);
      values[DIM * 2 + i] = iclp(((x0 + x4) >> 14).cast(intermType), constM256, const255);
      values[DIM * 3 + i] = iclp(((x8 + x6) >> 14).cast(intermType), constM256, const255);
      values[DIM * 4 + i] = iclp(((x8 - x6) >> 14).cast(intermType), constM256, const255);
      values[DIM * 5 + i] = iclp(((x0 - x4) >> 14).cast(intermType), constM256, const255);
      values[DIM * 6 + i] = iclp(((x3 - x2) >> 14).cast(intermType), constM256, const255);
      values[DIM * 7 + i] = iclp(((x7 - x1) >> 14).cast(intermType), constM256, const255);
    }

    const DFType bitsType = dfRawBits(outElementType.getTotalBits());
    DFVariable currConcat = values[0].cast(bitsType);
    for (unsigned i = 1; i < SIZE; ++i) {
      DFVariable casted = values[i].cast(bitsType);
      currConcat = casted.cat(currConcat);
    }

    assert(currConcat.getTotalBits() == outType.getTotalBits());

    DFVariable out = io.output("out", outType);
    out.connect(currConcat.cast(outType));
  }
};
