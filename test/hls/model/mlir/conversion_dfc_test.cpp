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
using Library = eda::hls::library::Library;
using Mapper = eda::hls::mapper::Mapper;
template<typename Type>
using ParametersOptimizer = eda::hls::scheduler::ParametersOptimizer<Type>;
using TopologicalBalancer = eda::hls::scheduler::TopologicalBalancer;
template<typename Type>
using Transformer = mlir::transforms::Transformer<Type>;

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

int conversionDfcTest(const dfc::kernel &kernel) {
  const std::string funcName = kernel.name;

  auto &builder = Builder::get();
  std::shared_ptr<Model> model = builder.create(funcName, funcName);

  uassert(model != nullptr, "Could not build model for kernel " + funcName +
                            "!\n");

  // Print initial model.
  std::cout << "Initial model:" << std::endl;
  std::cout << *model << std::endl;

  // Applying conversion in MLIR.
  Transformer<Model> transformer{*model};
  transformer.addPass(createHILToFIRRTLPass());
  transformer.runPasses();
  transformer.clearPasses();

  return 0;
}

TEST(ConversionDfcTest, ConversionDfcTestVectorSum) {
  std::shared_ptr<VectorSum> kernel = DFC_CREATE_KERNEL(VectorSum);

  EXPECT_EQ(conversionDfcTest(*kernel), 0);
}