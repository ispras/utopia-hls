//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

#include "gtest/gtest.h"

class MatrixMul2 : public dfcxx::Kernel {
public:
  std::string_view getName() override {
    return "MatrixMul2";
  }

  ~MatrixMul2() override = default;

  MatrixMul2() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType type = dfUInt(32);
    DFVariable x11 = io.input("x11", type);
    DFVariable x12 = io.input("x12", type);
    DFVariable x21 = io.input("x21", type);
    DFVariable x22 = io.input("x22", type);
    DFVariable y11 = io.input("y11", type);
    DFVariable y12 = io.input("y12", type);
    DFVariable y21 = io.input("y21", type);
    DFVariable y22 = io.input("y22", type);

    DFVariable mul11 = x11 * y11;
    DFVariable mul12 = x12 * y21;
    DFVariable mul21 = x11 * y12;
    DFVariable mul22 = x12 * y22;
    DFVariable mul31 = x21 * y11;
    DFVariable mul32 = x22 * y21;
    DFVariable mul41 = x21 * y12;
    DFVariable mul42 = x22 * y22;

    DFVariable sum1 = mul11 + mul12;
    DFVariable sum2 = mul21 + mul22;
    DFVariable sum3 = mul31 + mul32;
    DFVariable sum4 = mul41 + mul42;

    DFVariable out11 = io.output("out11", type);
    out11.connect(sum1);
    DFVariable out12 = io.output("out12", type);
    out12.connect(sum2);
    DFVariable out21 = io.output("out21", type);
    out21.connect(sum3);
    DFVariable out22 = io.output("out22", type);
    out22.connect(sum4);
  }
};

TEST(DFCxx, MatrixMul2_add_int_2_mul_int_3_ASAP) {
  MatrixMul2 kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2},
          {dfcxx::MUL_INT, 3},
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::ASAP), true);
}

TEST(DFCxx, MatrixMul2_add_int_2_mul_int_3_Linear) {
  MatrixMul2 kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2},
          {dfcxx::MUL_INT, 2},
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}