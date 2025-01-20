//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

static const int32_t kDIM = 8;
static const int32_t kSIZE = kDIM * kDIM;

class IDCT : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "IDCT";
  }

  ~IDCT() override = default;

  using DFType = dfcxx::DFType;
  using DFVariable = dfcxx::DFVariable;

  DFVariable iclp(DFVariable var, DFVariable left, DFVariable right) {
    const DFType muxType = dfInt(4);
    DFVariable firstRange = (var >= left).cast(muxType);
    DFVariable secondRange = (var > right).cast(muxType);
    DFVariable muxValue = firstRange + secondRange;
    return control.mux(muxValue, {left, var, right});
  }

  IDCT() : dfcxx::Kernel() {
    const DFType shortType = dfInt(16);
    const DFType type = dfInt(32);
    DFVariable const128 = constant.var(type, int64_t(128));
    DFVariable const181 = constant.var(type, int64_t(181));
    DFVariable const8192 = constant.var(type, int64_t(8192));
    DFVariable const4 = constant.var(type, int64_t(4));
    
    DFVariable constM256 = constant.var(shortType, int64_t(-256));
    DFVariable const255 = constant.var(shortType, int64_t(255));
    
    DFVariable W1 = constant.var(type, int64_t(2841));
    DFVariable W2 = constant.var(type, int64_t(2676));
    DFVariable W3 = constant.var(type, int64_t(2408));
    DFVariable W5 = constant.var(type, int64_t(1609));
    DFVariable W6 = constant.var(type, int64_t(1108));
    DFVariable W7 = constant.var(type, int64_t(565));
    
    std::vector<DFVariable> values;
    
    for (unsigned i = 0; i < kSIZE; ++i) {
      values.push_back(io.input("x" + std::to_string(i), shortType));
    }
    
    for (unsigned i = 0; i < kDIM; ++i) {
      DFVariable x0 = ((values[kDIM * i + 0]).cast(type) << 11) + const128;
      DFVariable x1 = ((values[kDIM * i + 4]).cast(type) << 11);
      DFVariable x2 = values[kDIM * i + 6].cast(type);
      DFVariable x3 = values[kDIM * i + 2].cast(type);
      DFVariable x4 = values[kDIM * i + 1].cast(type);
      DFVariable x5 = values[kDIM * i + 7].cast(type);
      DFVariable x6 = values[kDIM * i + 5].cast(type);
      DFVariable x7 = values[kDIM * i + 3].cast(type);
      
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
      
      values[kDIM * i + 0] = ((x7 + x1) >> 8).cast(shortType);
      values[kDIM * i + 1] = ((x3 + x2) >> 8).cast(shortType);
      values[kDIM * i + 2] = ((x0 + x4) >> 8).cast(shortType);
      values[kDIM * i + 3] = ((x8 + x6) >> 8).cast(shortType);
      values[kDIM * i + 4] = ((x8 - x6) >> 8).cast(shortType);
      values[kDIM * i + 5] = ((x0 - x4) >> 8).cast(shortType);
      values[kDIM * i + 6] = ((x3 - x2) >> 8).cast(shortType);
      values[kDIM * i + 7] = ((x7 - x1) >> 8).cast(shortType);
    }
    
    for (unsigned i = 0; i < kDIM; ++i) {
      DFVariable x0 = (values[kDIM * 0 + i].cast(type) << 8) + const8192;
      DFVariable x1 = (values[kDIM * 4 + i].cast(type) << 8);
      DFVariable x2 = values[kDIM * 6 + i].cast(type);
      DFVariable x3 = values[kDIM * 2 + i].cast(type);
      DFVariable x4 = values[kDIM * 1 + i].cast(type);
      DFVariable x5 = values[kDIM * 7 + i].cast(type);
      DFVariable x6 = values[kDIM * 5 + i].cast(type);
      DFVariable x7 = values[kDIM * 3 + i].cast(type);
      
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

      values[kDIM * 0 + i] = iclp(((x7 + x1) >> 14).cast(shortType), constM256, const255);
      values[kDIM * 1 + i] = iclp(((x3 + x2) >> 14).cast(shortType), constM256, const255);
      values[kDIM * 2 + i] = iclp(((x0 + x4) >> 14).cast(shortType), constM256, const255);
      values[kDIM * 3 + i] = iclp(((x8 + x6) >> 14).cast(shortType), constM256, const255);
      values[kDIM * 4 + i] = iclp(((x8 - x6) >> 14).cast(shortType), constM256, const255);
      values[kDIM * 5 + i] = iclp(((x0 - x4) >> 14).cast(shortType), constM256, const255);
      values[kDIM * 6 + i] = iclp(((x3 - x2) >> 14).cast(shortType), constM256, const255);
      values[kDIM * 7 + i] = iclp(((x7 - x1) >> 14).cast(shortType), constM256, const255);
    }

    for (unsigned i = 0; i < kSIZE; ++i) {
      DFVariable out = io.output("out" + std::to_string(i), shortType);
      out.connect(values[i]);
    } 
  }
};
