//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

using dfcxx::DFType;
using dfcxx::DFVariable;

class MagmaEncoder : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "MagmaEncoder";
  }

  ~MagmaEncoder() override = default;

  using DFType = dfcxx::DFType;
  using DFVariable = dfcxx::DFVariable;

  DFVariable magmaPermut(int tableId, DFVariable val) {
    const DFType type = dfUInt(4);

    DFVariable c0 = constant.var(type, uint64_t(0));
    DFVariable c1 = constant.var(type, uint64_t(1));
    DFVariable c2 = constant.var(type, uint64_t(2));
    DFVariable c3 = constant.var(type, uint64_t(3));
    DFVariable c4 = constant.var(type, uint64_t(4));
    DFVariable c5 = constant.var(type, uint64_t(5));
    DFVariable c6 = constant.var(type, uint64_t(6));
    DFVariable c7 = constant.var(type, uint64_t(7));
    DFVariable c8 = constant.var(type, uint64_t(8));
    DFVariable c9 = constant.var(type, uint64_t(9));
    DFVariable c10 = constant.var(type, uint64_t(10));
    DFVariable c11 = constant.var(type, uint64_t(11));
    DFVariable c12 = constant.var(type, uint64_t(12));
    DFVariable c13 = constant.var(type, uint64_t(13));
    DFVariable c14 = constant.var(type, uint64_t(14));
    DFVariable c15 = constant.var(type, uint64_t(15));

    switch (tableId) {
      case 0: {
        return control.mux(val, {
            c12, c4, c6, c2,
            c10, c5, c11, c9,
            c14, c8, c13, c7,
            c0, c3, c15, c1
        });
      }
      case 1: {
        return control.mux(val, {
            c6, c8, c2, c3,
            c9, c10, c5, c12,
            c1, c14, c4, c7,
            c11, c13, c0, c15
        });
      }
      case 2: {
        return control.mux(val, {
            c11, c3, c5, c8,
            c2, c15, c10, c13,
            c14, c1, c7, c4,
            c12, c9, c6, c0
        });
      }
      case 3: {
        return control.mux(val, {
            c12, c8, c2, c1,
            c13, c4, c15, c6,
            c7, c0, c10, c5,
            c3, c14, c9, c11
        });
      }
      case 4: {
        return control.mux(val, {
            c7, c15, c5, c10,
            c8, c1, c6, c13,
            c0, c9, c3, c14,
            c11, c4, c2, c12
        });
      }
      case 5: {
        return control.mux(val, {
            c5, c13, c15, c6,
            c9, c2, c12, c10,
            c11, c7, c8, c1,
            c4, c3, c14, c0
        });
      }
      case 6: {
        return control.mux(val, {
            c8, c14, c2, c5,
            c6, c9, c1, c12,
            c15, c4, c11, c0,
            c13, c10, c3, c7
        });
      }
      case 7: {
        return control.mux(val, {
            c1, c7, c14, c13,
            c0, c5, c8, c3,
            c4, c15, c10, c6,
            c9, c12, c11, c2
        });
      }
      default: return c0;
    }
  }
  
  DFVariable magmaIter(DFVariable left, DFVariable right, DFVariable key) {
    const DFType type = dfUInt(32);
    
    DFVariable sum = right.cast(type) + key.cast(type);
    DFVariable substituted = magmaPermut(7, sum(31, 28));
    for (int i = 1; i < 8; ++i) {
      int currSInd = 31 - i*4;
      substituted = magmaPermut(7 - i, sum(currSInd, currSInd - 3)).cat(substituted);
    }
    DFVariable shifted = substituted(20, 0).cat(substituted(31, 21));
    return left ^ shifted;
  }

  MagmaEncoder() : dfcxx::Kernel() {
    const DFType ioType = dfUInt(64);

    DFVariable block = io.input("block", ioType);
    DFVariable key = io.input("key", dfUInt(256));

    DFVariable currLeft = block(63, 32);
    DFVariable currRight = block(31, 0);

    for (int i = 0; i < 2; ++i) {
      for (int kInd = 0; kInd < 7; ++kInd) {
        int currKInd = 255 - 32*kInd;
        DFVariable buf = currRight;
        currRight = magmaIter(currLeft, currRight, key(currKInd, currKInd - 31));
        currLeft = buf;
      }
    }

    for (int kInd = 7; kInd >= 1; --kInd) {
      int currKInd = 255 - 32*kInd;
      DFVariable buf = currRight;
      currRight = magmaIter(currLeft, currRight, key(currKInd, currKInd - 31));
      currLeft = buf;
    }

    currLeft = magmaIter(currLeft, currRight, key(255, 224));

    DFVariable encoded = io.output("encoded", ioType);
    encoded.connect(currLeft.cat(currRight));
  }
};
