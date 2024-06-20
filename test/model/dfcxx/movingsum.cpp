#include "dfcxx/DFCXX.h"

#include "gtest/gtest.h"

class MovingSum : public dfcxx::Kernel {
public:
  std::string_view getName() override {
    return "MovingSum";
  }

  ~MovingSum() override = default;

  MovingSum() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType &type = dfUInt(32);
    DFVariable x = io.input("x", type);
    DFVariable x_minus = offset(x, -1);
    DFVariable x_plus = offset(x, 1);
    DFVariable sum1 = x_minus + x;
    DFVariable sum2 = sum1 + x_plus;
    DFVariable out = io.output("out", type);
    out.connect(sum2);
  }

};

TEST(DFCxx, MovingSum_add_int_2_mul_int_3_ASAP) {
  MovingSum kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2},
          {dfcxx::MUL_INT, 3}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Dijkstra), true);
}

TEST(DFCxx, MovingSum_add_int_2_mul_int_3_Linear) {
  MovingSum kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2},
          {dfcxx::MUL_INT, 3}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}

TEST(DFCxx, MovingSum_add_int_8_mul_int_15_ASAP) {
  MovingSum kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 8},
          {dfcxx::MUL_INT, 15}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Dijkstra), true);
}

TEST(DFCxx, MovingSum_add_int_8_mul_int_15_Linear) {
  MovingSum kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 8},
          {dfcxx::MUL_INT, 15}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}