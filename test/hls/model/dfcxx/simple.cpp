#include "dfcxx/DFCXX.h"

#include "gtest/gtest.h"

class SimpleKernel : public dfcxx::Kernel {
public:
  std::string_view getName() override {
    return "SimpleKernel";
  }

  ~SimpleKernel() override = default;

  SimpleKernel() : dfcxx::Kernel() {
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

TEST(DFCxxTest, Simple_1_Dijkstra) {
  SimpleKernel kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2},
          {dfcxx::MUL_INT, 3}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Dijkstra), true);
}

TEST(DFCxxTest, Simple_1_Linear) {
  SimpleKernel kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2},
          {dfcxx::MUL_INT, 3}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}

TEST(DFCxxTest, Simple_2_Dijkstra) {
  SimpleKernel kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 8},
          {dfcxx::MUL_INT, 15}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Dijkstra), true);
}

TEST(DFCxxTest, Simple_2_Linear) {
  SimpleKernel kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 8},
          {dfcxx::MUL_INT, 15}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}