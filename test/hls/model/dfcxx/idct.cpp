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
    DFVariable const256 = constant.var(type, int64_t(256));
    DFVariable const181 = constant.var(type, int64_t(181));
    DFVariable const8 = constant.var(type, int64_t(8));
    DFVariable const8192 = constant.var(type, int64_t(8192));
    DFVariable const4 = constant.var(type, int64_t(4));
    DFVariable const16384 = constant.var(type, int64_t(16384));
    
    DFVariable W1 = constant.var(type, int64_t(2841));
    DFVariable W2 = constant.var(type, int64_t(2676));
    DFVariable W3 = constant.var(type, int64_t(2408));
    DFVariable W5 = constant.var(type, int64_t(1609));
    DFVariable W6 = constant.var(type, int64_t(1108));
    DFVariable W7 = constant.var(type, int64_t(565));
    
    std::vector<DFVariable *> inputs(kSIZE);
    std::vector<DFVariable *> buf(kSIZE);
    std::vector<DFVariable *> values(kSIZE);
    
    for (unsigned i = 0; i < kSIZE; ++i) {
      inputs[i] = io.input("x" + std::to_string(i), type);
    }
    
    for (unsigned i = 0; i < kDIM; ++i) {
      DFVariable x0 = (*(inputs[kDIM * i + 0]) << 11) + const128;
      DFVariable x1 = (*(inputs[kDIM * i + 4]) << 11);
      DFVariable x2 = *(inputs[kDIM * i + 6]);
      DFVariable x3 = *(inputs[kDIM * i + 2]);
      DFVariable x4 = *(inputs[kDIM * i + 1]);
      DFVariable x5 = *(inputs[kDIM * i + 7]);
      DFVariable x6 = *(inputs[kDIM * i + 5]);
      DFVariable x7 = *(inputs[kDIM * i + 3]);
      
      DFVariable x8 = (x4 + x5) * W7;
      DFVariable x4_2 = x8 + (W1 - W7) * x4;
      DFVariable x5_2 = x8 - (W1 - W7) * x5;
      DFVariable x8_2 = W3 * (x6 + x7);
      DFVariable x6_2 = x8_2 - (W3 - W5) * x6;
      DFVariable x7_2 = x8_2 - (W3 + W5) * x7;
      
      DFVariable x8_3 = x0 + x1;
      DFVariable x0_2 = x0 - x1;
      DFVariable x1_2 = (x3 + x2) * W6;
      DFVariable x2_2 = x1_2 - (W2 + W6) * x2;
      DFVariable x3_2 = x1_2 + (W2 - W6) * x3;
      DFVariable x1_3 = x4_2 + x6_2;
      DFVariable x4_3 = x4_2 - x6_2;
      DFVariable x6_3 = x5_2 + x7_2;
      DFVariable x5_3 = x5_2 - x7_2;
      
      DFVariable x7_3 = x8_3 + x3_2;
      DFVariable x8_4 = x8_3 - x3_2;
      DFVariable x3_3 = x0_2 + x2_2;
      DFVariable x0_3 = x0_2 - x2_2;
      DFVariable x2_3 = (((x4_3 + x5_3) * const181) + const128) >> 8;
      DFVariable x4_4 = (((x4_3 - x5_3) * const181) + const128) >> 8;
      
      buf[kDIM * i + 0] = &((x7_3 + x1_3) >> 8);
      buf[kDIM * i + 1] = &((x3_3 + x2_3) >> 8);
      buf[kDIM * i + 2] = &((x0_3 + x4_4) >> 8);
      buf[kDIM * i + 3] = &((x8_4 + x6_3) >> 8);
      buf[kDIM * i + 4] = &((x8_4 - x6_3) >> 8);
      buf[kDIM * i + 5] = &((x0_3 - x4_4) >> 8);
      buf[kDIM * i + 6] = &((x3_3 - x2_3) >> 8);
      buf[kDIM * i + 7] = &((x7_3 - x1_3) >> 8);
    }
    
    for (unsigned i = 0; i < kDIM; ++i) {
      DFVariable &x0 = (*(buf[kDIM * 0 + i]) << 8) + const8192;
      DFVariable &x1 = (*(buf[kDIM * 4 + i]) << 8);
      DFVariable &x2 = *(buf[kDIM * 6 + i]);
      DFVariable &x3 = *(buf[kDIM * 2 + i]);
      DFVariable &x4 = *(buf[kDIM * 1 + i]);
      DFVariable &x5 = *(buf[kDIM * 7 + i]);
      DFVariable &x6 = *(buf[kDIM * 5 + i]);
      DFVariable &x7 = *(buf[kDIM * 3 + i]);
      
      DFVariable &x8 = ((x4 + x5) * W7) + const4;
      DFVariable &x4_2 = (x8 + (W1 - W7) * x4) >> 3;
      DFVariable &x5_2 = (x8 - (W1 - W7) * x5) >> 3;
      DFVariable &x8_2 = (W3 * (x6 + x7)) + const4;
      DFVariable &x6_2 = (x8_2 - (W3 - W5) * x6) >> 3;
      DFVariable &x7_2 = (x8_2 - (W3 + W5) * x7) >> 3;
      
      DFVariable &x8_3 = x0 + x1;
      DFVariable &x0_2 = x0 - x1;
      DFVariable &x1_2 = ((x3 + x2) * W6) + const4;
      DFVariable &x2_2 = (x1_2 - (W2 + W6) * x2) >> 3;
      DFVariable &x3_2 = (x1_2 + (W2 - W6) * x3) >> 3;
      DFVariable &x1_3 = x4_2 + x6_2;
      DFVariable &x4_3 = x4_2 - x6_2;
      DFVariable &x6_3 = x5_2 + x7_2;
      DFVariable &x5_3 = x5_2 - x7_2;
      
      DFVariable &x7_3 = x8_3 + x3_2;
      DFVariable &x8_4 = x8_3 - x3_2;
      DFVariable &x3_3 = x0_2 + x2_2;
      DFVariable &x0_3 = x0_2 - x2_2;
      DFVariable &x2_3 = (((x4_3 + x5_3) * const181) + const128) >> 8;
      DFVariable &x4_4 = (((x4_3 - x5_3) * const181) + const128) >> 8;
      
      values[kDIM * 0 + i] = &((x7_3 + x1_3) >> 14);
      values[kDIM * 1 + i] = &((x3_3 + x2_3) >> 14);
      values[kDIM * 2 + i] = &((x0_3 + x4_4) >> 14);
      values[kDIM * 3 + i] = &((x8_4 + x6_3) >> 14);
      values[kDIM * 4 + i] = &((x8_4 - x6_3) >> 14);
      values[kDIM * 5 + i] = &((x0_3 - x4_4) >> 14);
      values[kDIM * 6 + i] = &((x3_3 - x2_3) >> 14);
      values[kDIM * 7 + i] = &((x7_3 - x1_3) >> 14);
    }
    
    for (unsigned i = 0; i < kSIZE; ++i) {
      DFVariable &out = io.output("out" + std::to_string(i), type);
      out.connect(*(values[i]));
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
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Dijkstra), true);
}
