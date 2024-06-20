#include "dfcxx/DFCXX.h"

#include "gtest/gtest.h"

class Polynomial2 : public dfcxx::Kernel {
public:
  std::string_view getName() override {
    return "Polynomial2";
  }

  ~Polynomial2() override = default;

  Polynomial2() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType &type = dfUInt(32);
    DFVariable x = io.input("x", type);
    DFVariable square = x * x;
    DFVariable result = square + x;
    DFVariable test = result + x;
    DFVariable out = io.output("out", type);
    out.connect(test);
  }
};

TEST(DFCxx, Polynomial2_add_int_2_mul_int_3_ASAP) {
  Polynomial2 kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2},
          {dfcxx::MUL_INT, 3}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::ASAP), true);
}

TEST(DFCxx, Polynomial2_add_int_2_mul_int_3_Linear) {
  Polynomial2 kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2},
          {dfcxx::MUL_INT, 3}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}

TEST(DFCxx, Polynomial2_add_int_8_mul_int_15_ASAP) {
  Polynomial2 kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 8},
          {dfcxx::MUL_INT, 15}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::ASAP), true);
}

TEST(DFCxx, Polynomial2_add_int_8_mul_int_15_Linear) {
  Polynomial2 kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 8},
          {dfcxx::MUL_INT, 15}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}