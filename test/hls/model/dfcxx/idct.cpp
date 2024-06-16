#include "dfcxx/DFCXX.h"

#include "gtest/gtest.h"

static const int32_t kDIM = 8;
static const int32_t kSIZE = kDIM * kDIM;

class IDCT : public dfcxx::Kernel {
public:
  std::string_view getName() override {
    return "IDCT";
  }

  ~IDCT() override = default;

  IDCT() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;
  
    const DFType type = dfInt(32);
    DFVariable const128 = constant.var(type, int64_t(128));
    DFVariable const181 = constant.var(type, int64_t(181));
    DFVariable const8192 = constant.var(type, int64_t(8192));
    DFVariable const4 = constant.var(type, int64_t(4));
    
    DFVariable W1 = constant.var(type, int64_t(2841));
    DFVariable W2 = constant.var(type, int64_t(2676));
    DFVariable W3 = constant.var(type, int64_t(2408));
    DFVariable W5 = constant.var(type, int64_t(1609));
    DFVariable W6 = constant.var(type, int64_t(1108));
    DFVariable W7 = constant.var(type, int64_t(565));
    
    std::vector<DFVariable> values;
    
    for (unsigned i = 0; i < kSIZE; ++i) {
      values.push_back(io.input("x" + std::to_string(i), type));
    }
    
    for (unsigned i = 0; i < kDIM; ++i) {
      DFVariable x0 = ((values[kDIM * i + 0]) << 11) + const128;
      DFVariable x1 = ((values[kDIM * i + 4]) << 11);
      DFVariable x2 = (values[kDIM * i + 6]);
      DFVariable x3 = (values[kDIM * i + 2]);
      DFVariable x4 = (values[kDIM * i + 1]);
      DFVariable x5 = (values[kDIM * i + 7]);
      DFVariable x6 = (values[kDIM * i + 5]);
      DFVariable x7 = (values[kDIM * i + 3]);
      
      DFVariable x8 = (x4 + x5) * W7;
      x4 = x8 + (W1 - W7) * x4;
      x5 = x8 - (W1 - W7) * x5;
      x8 = W3 * (x6 + x7);
      x6 = x8 - (W3 - W5) * x6;
      x7 = x8 - (W3 + W5) * x7;
      
      x8 = x0 + x1;
      x0 = x0 - x1;
      x1 = (x3 + x2) * W6;
      x2 = x1 - (W2 + W6) * x2;
      x3 = x1 + (W2 - W6) * x3;
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
      
      values[kDIM * i + 0] = (x7 + x1) >> 8;
      values[kDIM * i + 1] = (x3 + x2) >> 8;
      values[kDIM * i + 2] = (x0 + x4) >> 8;
      values[kDIM * i + 3] = (x8 + x6) >> 8;
      values[kDIM * i + 4] = (x8 - x6) >> 8;
      values[kDIM * i + 5] = (x0 - x4) >> 8;
      values[kDIM * i + 6] = (x3 - x2) >> 8;
      values[kDIM * i + 7] = (x7 - x1) >> 8;
    }
    
    for (unsigned i = 0; i < kDIM; ++i) {
      DFVariable x0 = (values[kDIM * 0 + i] << 8) + const8192;
      DFVariable x1 = (values[kDIM * 4 + i] << 8);
      DFVariable x2 = values[kDIM * 6 + i];
      DFVariable x3 = values[kDIM * 2 + i];
      DFVariable x4 = values[kDIM * 1 + i];
      DFVariable x5 = values[kDIM * 7 + i];
      DFVariable x6 = values[kDIM * 5 + i];
      DFVariable x7 = values[kDIM * 3 + i];
      
      DFVariable x8 = ((x4 + x5) * W7) + const4;
      x4 = (x8 + (W1 - W7) * x4) >> 3;
      x5 = (x8 - (W1 - W7) * x5) >> 3;
      x8 = (W3 * (x6 + x7)) + const4;
      x6 = (x8 - (W3 - W5) * x6) >> 3;
      x7 = (x8 - (W3 + W5) * x7) >> 3;
      
      x8 = x0 + x1;
      x0 = x0 - x1;
      x1 = ((x3 + x2) * W6) + const4;
      x2 = (x1 - (W2 + W6) * x2) >> 3;
      x3 = (x1 + (W2 - W6) * x3) >> 3;
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
      
      values[kDIM * 0 + i] = (x7 + x1) >> 14;
      values[kDIM * 1 + i] = (x3 + x2) >> 14;
      values[kDIM * 2 + i] = (x0 + x4) >> 14;
      values[kDIM * 3 + i] = (x8 + x6) >> 14;
      values[kDIM * 4 + i] = (x8 - x6) >> 14;
      values[kDIM * 5 + i] = (x0 - x4) >> 14;
      values[kDIM * 6 + i] = (x3 - x2) >> 14;
      values[kDIM * 7 + i] = (x7 - x1) >> 14;
    }
    
    for (unsigned i = 0; i < kSIZE; ++i) {
      DFVariable out = io.output("out" + std::to_string(i), type);
      out.connect(values[i]);
    }  
  }
};

TEST(DFCxxTest, IDCT_1_Dijkstra) {
  IDCT kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 1},
          {dfcxx::MUL_INT, 3},
          {dfcxx::SUB_INT, 1}
  };
  EXPECT_EQ(kernel.compile(config, dfcxx::Dijkstra), true);
}
