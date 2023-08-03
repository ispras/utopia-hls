//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that reads *.mlir file, applies graph
// rewriting pass on the program representation and stores the result back.
//
//===----------------------------------------------------------------------===//

#include "HIL/Combine.h"
#include "HIL/Dialect.h"
#include "HIL/Model.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;

using MLIRContext = mlir::MLIRContext;
using ModuleOp = mlir::ModuleOp;
template<typename Type>
using LLVMErrorOr = llvm::ErrorOr<Type>;
using LLVMMemoryBuffer = llvm::MemoryBuffer;
using LLVMSourceMgr = llvm::SourceMgr;
template<typename Type>
using OwningOpRef = mlir::OwningOpRef<Type>;
using PassManager = mlir::PassManager;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

int loadMLIR(MLIRContext &context,
             OwningOpRef<ModuleOp> &module) {
  // Open file.
  LLVMErrorOr<std::unique_ptr<LLVMMemoryBuffer>> fileOrErr =
      LLVMMemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code err = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << err.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  LLVMSourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "hil dialect");

  MLIRContext context;

  context.getOrLoadDialect<mlir::hil::HILDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();

  OwningOpRef<ModuleOp> module;

  if (int error = loadMLIR(context, module))
    return error;

  PassManager pm(&context);
  mlir::applyPassManagerCLOptions(pm);
  pm.addPass(createGraphRewritePass());

  if (mlir::failed(pm.run(*module)))
    return 4;

  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
      llvm::errs() << errorMessage << "\n";
      return 5;
  }

  module->print(output->os());
  return 0;
}