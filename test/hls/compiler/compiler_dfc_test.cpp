//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/compiler/compiler.h"
#include "hls/library/library.h"
#include "hls/mapper/mapper.h"
#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/dfc/dfc.h"
#include "hls/parser/dfc/internal/builder.h"
#include "hls/scheduler/param_optimizer.h"
#include "hls/scheduler/topological_balancer.h"

#include "gtest/gtest.h"

#include <array>

using namespace eda::hls::compiler;
using namespace eda::hls::library;
using namespace eda::hls::scheduler;
using namespace eda::hls::mapper;

DFC_KERNEL(IDCT) {
  static const int W1 = 2841; // 2048*sqrt(2)*cos(1*pi/16)
  static const int W2 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
  static const int W3 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
  static const int W5 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
  static const int W6 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
  static const int W7 = 565;  // 2048*sqrt(2)*cos(7*pi/16)

  DFC_KERNEL_CTOR(IDCT) {
    std::array<dfc::stream<dfc::sint16>, 64> blk;

    for (std::size_t i = 0; i < 8; i++)
      idctrow(blk, i);

    for (std::size_t i = 0; i < 8; i++)
      idctcol(blk, i);
  }

  /* row (horizontal) IDCT
   *
   *           7                       pi         1
   * dst[k] = sum c[l] * src[l] * cos( -- * ( k + - ) * l )
   *          l=0                      8          2
   *
   * where: c[0]    = 128
   *        c[1..7] = 128*sqrt(2)
   */
  void idctrow(std::array<dfc::stream<dfc::sint16>, 64> &blk, std::size_t i) {
    dfc::stream<dfc::sint32> x0, x1, x2, x3, x4, x5, x6, x7, x8;

    /* TODO: shortcut */
    x1 = dfc::cast<dfc::sint32>(blk[8*i+4])<<11;
    x2 = dfc::cast<dfc::sint32>(blk[8*i+6]);
    x3 = dfc::cast<dfc::sint32>(blk[8*i+2]);
    x4 = dfc::cast<dfc::sint32>(blk[8*i+1]);
    x5 = dfc::cast<dfc::sint32>(blk[8*i+7]);
    x6 = dfc::cast<dfc::sint32>(blk[8*i+5]);
    x7 = dfc::cast<dfc::sint32>(blk[8*i+3]);

    x0 = (dfc::cast<dfc::sint32>(blk[0])<<11) + 128; /* for proper rounding in the fourth stage */

    /* first stage */
    x8 = W7*(x4+x5);
    x4 = x8 + (W1-W7)*x4;
    x5 = x8 - (W1+W7)*x5;
    x8 = W3*(x6+x7);
    x6 = x8 - (W3-W5)*x6;
    x7 = x8 - (W3+W5)*x7;

    /* second stage */
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6*(x3+x2);
    x2 = x1 - (W2+W6)*x2;
    x3 = x1 + (W2-W6)*x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    /* third stage */
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181*(x4+x5)+128)>>8;
    x4 = (181*(x4-x5)+128)>>8;

    /* fourth stage */
    blk[0] = dfc::cast<dfc::sint16>((x7+x1)>>8);
    blk[1] = dfc::cast<dfc::sint16>((x3+x2)>>8);
    blk[2] = dfc::cast<dfc::sint16>((x0+x4)>>8);
    blk[3] = dfc::cast<dfc::sint16>((x8+x6)>>8);
    blk[4] = dfc::cast<dfc::sint16>((x8-x6)>>8);
    blk[5] = dfc::cast<dfc::sint16>((x0-x4)>>8);
    blk[6] = dfc::cast<dfc::sint16>((x3-x2)>>8);
    blk[7] = dfc::cast<dfc::sint16>((x7-x1)>>8);
  }

  /* column (vertical) IDCT
   *
   *             7                         pi         1
   * dst[8*k] = sum c[l] * src[8*l] * cos( -- * ( k + - ) * l )
   *            l=0                        8          2
   *
   * where: c[0]    = 1/1024
   *        c[1..7] = (1/1024)*sqrt(2)
   */
  void idctcol(std::array<dfc::stream<dfc::sint16>, 64> blk, std::size_t i) {
    dfc::stream<dfc::sint32> x0, x1, x2, x3, x4, x5, x6, x7, x8;

    /* TODO: shortcut */
    x1 = dfc::cast<dfc::sint32>(blk[8*4+i])<<8;
    x2 = dfc::cast<dfc::sint32>(blk[8*6+i]);
    x3 = dfc::cast<dfc::sint32>(blk[8*2+i]);
    x4 = dfc::cast<dfc::sint32>(blk[8*1+i]);
    x5 = dfc::cast<dfc::sint32>(blk[8*7+i]);
    x6 = dfc::cast<dfc::sint32>(blk[8*5+i]);
    x7 = dfc::cast<dfc::sint32>(blk[8*3+i]);

    x0 = (dfc::cast<dfc::sint32>(blk[8*0+i])<<8) + 8192;

    /* first stage */
    x8 = W7*(x4+x5) + 4;
    x4 = (x8+(W1-W7)*x4)>>3;
    x5 = (x8-(W1+W7)*x5)>>3;
    x8 = W3*(x6+x7) + 4;
    x6 = (x8-(W3-W5)*x6)>>3;
    x7 = (x8-(W3+W5)*x7)>>3;

    /* second stage */
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6*(x3+x2) + 4;
    x2 = (x1-(W2+W6)*x2)>>3;
    x3 = (x1+(W2-W6)*x3)>>3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    /* third stage */
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181*(x4+x5)+128)>>8;
    x4 = (181*(x4-x5)+128)>>8;

    /* fourth stage */
    blk[8*0+i] = iclp((x7+x1)>>14);
    blk[8*1+i] = iclp((x3+x2)>>14);
    blk[8*2+i] = iclp((x0+x4)>>14);
    blk[8*3+i] = iclp((x8+x6)>>14);
    blk[8*4+i] = iclp((x8-x6)>>14);
    blk[8*5+i] = iclp((x0-x4)>>14);
    blk[8*6+i] = iclp((x3-x2)>>14);
    blk[8*7+i] = iclp((x7-x1)>>14);
  }

  // (i<-256) ? -256 : ((i>255) ? 255 : i)
  dfc::typed<dfc::sint16>& iclp(dfc::typed<dfc::sint32> &i) {
    dfc::value<dfc::sint32> Cm256(-256);
    dfc::value<dfc::sint32> Cp255(+255);

    return dfc::cast<dfc::sint16>(dfc::mux(i < Cm256, Cm256, mux(i > Cp255, Cp255, i)));
  }
};

int compilerDfcTest(const std::string &inputLibraryPath,
                    const std::string &relativeCompPath,
                    const std::string &outputFirrtlName,
                    const std::string &outputVerilogLibraryName,
                    const std::string &outputVerilogTopModuleName,
                    const std::string &outputDirName,
                    const std::string &outputTestName) {

  Indicators indicators;
  // Optimization criterion and constraints.
  eda::hls::model::Criteria criteria(
    PERF,
    eda::hls::model::Constraint<unsigned>(40000, 500000),                                // Frequency (kHz)
    eda::hls::model::Constraint<unsigned>(1000,  500000),                                // Performance (=frequency)
    eda::hls::model::Constraint<unsigned>(0,     1000),                                  // Latency (cycles)
    eda::hls::model::Constraint<unsigned>(),                                             // Power (does not matter)
    eda::hls::model::Constraint<unsigned>(1,     10000000));

  dfc::params args;
  IDCT kernel(args);

  std::shared_ptr<Model> model =
    eda::hls::parser::dfc::Builder::get().create("IDCT");

  Library::get().initialize(inputLibraryPath, relativeCompPath);

  Mapper::get().map(*model, Library::get());
  std::map<std::string, Parameters> params =
    ParametersOptimizer<TopologicalBalancer>::get().optimize(criteria,
                                                             *model,
                                                             indicators);

  TopologicalBalancer::get().balance(*model);

  auto compiler = std::make_unique<Compiler>();
  auto circuit = compiler->constructFirrtlCircuit(*model, "IDCT");
  circuit->printFiles(outputFirrtlName,
                      outputVerilogLibraryName,
                      outputVerilogTopModuleName,
                      outputDirName);

  // generate random test of the specified length in ticks
  const int testLength = 10;
  circuit->printRndVlogTest(*model,
                            outputDirName,
                            outputTestName,
                            testLength);

  Library::get().finalize();
  return 0;
}

TEST(CompilerDfcTest, CompilerDfcTestIdct) {
  EXPECT_EQ(compilerDfcTest("./test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "outputIdctFirrtl.mlir",
                            "outputIdctVerilogLibrary.v",
                            "outputIdctVerilogTopModule.v",
                            "./output/test/dfc/idct",
                            "outputIdctTestbench.v"), 0);
}
