//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/DFCIROperations.h"

#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser/Parser.h"

#include <fstream>

#define ADDCONST_DATA_PATH TEST_EXAMPLES_PATH "/addconst/addconst.mlir"

TEST(ExamplesAddConst, DFCIRParsePrint) {
  std::ifstream fStream(ADDCONST_DATA_PATH);
  std::stringstream buf;
  buf << fStream.rdbuf();
  std::string inputDfcir = buf.str();

  // Parse the input DFCIR file.
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::dfcir::DFCIRDialect>();
  mlir::ParserConfig parserCfg(&ctx);
  auto module = mlir::parseSourceString(inputDfcir, parserCfg);
  ASSERT_NE(*module, nullptr);

  // Print the parsed DFCIR file.
  std::string parsedDfcir;
  llvm::raw_string_ostream stream(parsedDfcir);
  module->print(stream);

  // Compare the initial and parsed representations.
  ASSERT_STREQ(parsedDfcir.c_str(), inputDfcir.c_str());
}
