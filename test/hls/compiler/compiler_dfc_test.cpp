//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "HIL/API.h"
#include "HIL/Dumper.h"
#include "HIL/Model.h"
#include "hls/compiler/compiler.h"
#include "hls/library/library.h"
#include "hls/mapper/mapper.h"
#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"
#include "hls/parser/dfc/dfc.h"
#include "hls/parser/dfc/internal/builder.h"
#include "hls/scheduler/param_optimizer.h"
#include "hls/scheduler/topological_balancer.h"
#include "utils/string.h"

#include "gtest/gtest.h"

#include <array>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

using Builder = eda::hls::parser::dfc::Builder;
using Compiler = eda::hls::compiler::Compiler;
using Criteria = eda::hls::model::Criteria;
using Library = eda::hls::library::Library;
using Mapper = eda::hls::mapper::Mapper;
template<typename T>
using ParametersOptimizer = eda::hls::scheduler::ParametersOptimizer<T>;
using TopologicalBalancer = eda::hls::scheduler::TopologicalBalancer;

DFC_KERNEL(VectorSum) {
  static const int SIZE = 4;

  DFC_KERNEL_CTOR(VectorSum) {
    DFC_KERNEL_ACTIVATE;

    std::vector<dfc::stream<dfc::sint16>> lhs;
    for (std::size_t i = 0; i < SIZE; i++) {
      lhs.push_back(dfc::stream<dfc::sint16>(std::string("lhs") +
                                             "_" +
                                             std::to_string(i)));
    }
    std::vector<dfc::stream<dfc::sint16>> rhs;
    for (std::size_t i = 0; i < SIZE; i++) {
      rhs.push_back(dfc::stream<dfc::sint16>(std::string("rhs") +
                                             "_" +
                                             std::to_string(i)));
    }
    std::vector<dfc::stream<dfc::sint16>> res;
    for (std::size_t i = 0; i < SIZE; i++) {
      res.push_back(dfc::stream<dfc::sint16>(std::string("res") +
                                             "_" +
                                             std::to_string(i)));
    }

    for (std::size_t i = 0; i < SIZE; i++) {
      res[i] = lhs[i] + rhs[i];
    }

    DFC_KERNEL_DEACTIVATE;
  }

  DFC_CREATE_KERNEL_FUNCTION(VectorSum);
};

DFC_KERNEL(ScalarMul) {
  static const int SIZE = 3;

  DFC_KERNEL_CTOR(ScalarMul) {
    DFC_KERNEL_ACTIVATE;

    std::vector<dfc::stream<dfc::sint16>> lhs;
    for (std::size_t i = 0; i < SIZE; i++) {
      lhs.push_back(dfc::stream<dfc::sint16>(std::string("lhs") +
                                             "_" +
                                             std::to_string(i)));
    }
    std::vector<dfc::stream<dfc::sint16>> rhs;
    for (std::size_t i = 0; i < SIZE; i++) {
      rhs.push_back(dfc::stream<dfc::sint16>(std::string("rhs") +
                                             "_" +
                                             std::to_string(i)));
    }
    dfc::stream<dfc::sint16> res;

    res = lhs[0] * rhs[0]; 
    for (std::size_t i = 1; i < lhs.size(); i++) {
      res += lhs[i] * rhs[i];
    }

    DFC_KERNEL_DEACTIVATE;
  }

  DFC_CREATE_KERNEL_FUNCTION(ScalarMul);
};

DFC_KERNEL(IDCT) {
  static const int W1 = 2841; // 2048*sqrt(2)*cos(1*pi/16)
  static const int W2 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
  static const int W3 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
  static const int W5 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
  static const int W6 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
  static const int W7 = 565;  // 2048*sqrt(2)*cos(7*pi/16)

  DFC_KERNEL_CTOR(IDCT) {
    DFC_KERNEL_ACTIVATE;

    std::vector<dfc::stream<dfc::sint16>> blk;
    for (std::size_t i = 0; i < 64; i++) {
      blk.push_back(dfc::stream<dfc::sint16>(std::string("blk") +
                                             "_" +
                                             std::to_string(i)));
    }

    for (std::size_t i = 0; i < 8; i++) {
      idctrow(blk, i);
    }

    for (std::size_t i = 0; i < 8; i++) {
      idctcol(blk, i);
    }

    DFC_KERNEL_DEACTIVATE;
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
  void idctrow(std::vector<dfc::stream<dfc::sint16>> &blk, std::size_t i) {
    dfc::stream<dfc::sint32> x0, x1, x2, x3, x4, x5, x6, x7, x8;

    /* TODO: shortcut */
    x1 = dfc::cast<dfc::sint32>(blk[8*i+4])<<11;
    x2 = dfc::cast<dfc::sint32>(blk[8*i+6]);
    x3 = dfc::cast<dfc::sint32>(blk[8*i+2]);
    x4 = dfc::cast<dfc::sint32>(blk[8*i+1]);
    x5 = dfc::cast<dfc::sint32>(blk[8*i+7]);
    x6 = dfc::cast<dfc::sint32>(blk[8*i+5]);
    x7 = dfc::cast<dfc::sint32>(blk[8*i+3]);

    /* for proper rounding in the fourth stage */
    x0 = (dfc::cast<dfc::sint32>(blk[8*i+0])<<11) + 128;

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
    blk[0+i*8] = dfc::cast<dfc::sint16>((x7+x1)>>8);
    blk[1+i*8] = dfc::cast<dfc::sint16>((x3+x2)>>8);
    blk[2+i*8] = dfc::cast<dfc::sint16>((x0+x4)>>8);
    blk[3+i*8] = dfc::cast<dfc::sint16>((x8+x6)>>8);
    blk[4+i*8] = dfc::cast<dfc::sint16>((x8-x6)>>8);
    blk[5+i*8] = dfc::cast<dfc::sint16>((x0-x4)>>8);
    blk[6+i*8] = dfc::cast<dfc::sint16>((x3-x2)>>8);
    blk[7+i*8] = dfc::cast<dfc::sint16>((x7-x1)>>8);
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
  void idctcol(std::vector<dfc::stream<dfc::sint16>> &blk, std::size_t i) {
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

    return dfc::cast<dfc::sint16>(dfc::mux(i < Cm256,
                                           dfc::mux(i > Cp255, i, Cp255),
                                           Cm256));
  }

  DFC_CREATE_KERNEL_FUNCTION(IDCT);
};

int compilerDfcTest(const dfc::kernel &kernel,
                    const std::string &inLibSubPath,
                    const std::string &relCatPath,
                    const std::string &outFirName,
                    const std::string &outVlogLibName,
                    const std::string &outVlogTopName,
                    const std::string &outSubPath,
                    const std::string &outTestName) {
  const std::string funcName = kernel.name;
  const fs::path homePath = std::string(getenv("UTOPIA_HLS_HOME"));

  // Optimization criterion and constraints.
  Criteria criteria(
    eda::hls::model::PERF,
    // Frequency (kHz)
    eda::hls::model::Constraint<unsigned>(40000, 500000), 
    // Performance (=frequency)
    eda::hls::model::Constraint<unsigned>(1000, 500000),
    // Latency (cycles)
    eda::hls::model::Constraint<unsigned>(0, 1000),
    // Power (does not matter)
    eda::hls::model::Constraint<unsigned>(),
    // Area (number of LUTs)
    eda::hls::model::Constraint<unsigned>(1, 10000000));

  auto &builder = Builder::get();
  std::shared_ptr<Model> model = builder.create(funcName, funcName);

  uassert(model != nullptr, "Could not build model for kernel " + funcName +
                            "!\n");

  // Print initial model.
  std::cout << "Initial model:" << std::endl;
  std::cout << *model << std::endl;

  // Create output directory.
  const fs::path fsOutPath = homePath / outSubPath;
  fs::create_directories(fsOutPath);

  // Print model '.dot' representation to file.
  const std::string outDotFileName = "dfc_" + eda::utils::toLower(funcName) +
                                     "_test.dot";
  std::ofstream output(fsOutPath / outDotFileName);
  eda::hls::model::printDot(output, *model);
  output.close();

  /// TODO: Uncomment when the end-to-end tool path is ready.
  /*const std::string inLibPath = homePath / inLibSubPath;
  Library::get().initialize();
  Library::get().importLibrary(inLibPath, relCatPath);

  Mapper::get().map(*model, Library::get());
  auto &paramOptimizer = ParametersOptimizer<TopologicalBalancer>::get();

  Indicators indicators;
  auto params = paramOptimizer.optimize(criteria, *model, indicators);

  TopologicalBalancer::get().balance(*model);

  auto compiler = std::make_unique<Compiler>();
  
  uassert(compiler != nullptr, "Could not build hls compiler!");

  // Construct FIRRTL IR Circuit from the model
  auto circuit = compiler->constructFirrtlCircuit(*model, funcName);

  uassert(circuit != nullptr, "Could not build FIRRTL circuit from model " +
                              funcName + "!");

  const std::string outPath = fsOutPath;

  circuit->printFiles(outFirName, outVlogLibName, outVlogTopName, outPath);

  // Generate random test of the specified length in ticks
  const int testLength = 1;
  circuit->printRndVlogTest(*model, outPath, outTestName, testLength);

  std::string pathToOutVlogFiles = fsOutPath / "*.v";

  bool isCompiled = system(("iverilog " + pathToOutVlogFiles).c_str());

  Library::get().finalize();

  return isCompiled;*/
  return 0;
}

TEST(CompilerDfcTest, CompilerDfcTestIdct) {
  std::shared_ptr<IDCT> kernel = DFC_CREATE_KERNEL(IDCT);

  EXPECT_EQ(compilerDfcTest(*kernel,
                            "test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "idctFir.mlir",
                            "idctLib.v",
                            "idctTop.v",
                            "output/test/compiler/dfc/idct/",
                            "idctTestBench.v"), 0);
}

TEST(CompilerDfcTest, CompilerDfcTestVectorSum) {
  std::shared_ptr<VectorSum> kernel = DFC_CREATE_KERNEL(VectorSum);

  EXPECT_EQ(compilerDfcTest(*kernel,
                            "test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "vectorSumFir.mlir",
                            "vectorSumLib.v",
                            "vectorSumTop.v",
                            "output/test/compiler/dfc/vector_sum",
                            "vectorSumTestbench.v"), 0);
}

TEST(CompilerDfcTest, CompilerDfcTestScalarMul) {
  std::shared_ptr<ScalarMul> kernel = DFC_CREATE_KERNEL(ScalarMul);

  EXPECT_EQ(compilerDfcTest(*kernel,
                            "test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "scalarMulFir.mlir",
                            "scalarMulLib.v",
                            "scalarMulTop.v",
                            "output/test/compiler/dfc/scalar_mul/",
                            "scalarMulTestbench.v"), 0);
}