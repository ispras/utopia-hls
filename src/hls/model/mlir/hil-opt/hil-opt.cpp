//===- hil-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "HIL/HILCombine.h"
#include "HIL/HILDialect.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"), cl::value_desc("filename"));
static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
    // Otherwise, the input is '.mlir'.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    return 0;
}

int main(int argc, char **argv) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerTestPrintNestingPass();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "hil dialect");
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::hil::HILDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    mlir::OwningModuleRef module;

    if (int error = loadMLIR(context, module))
        return error;
    mlir::PassManager pm(&context);

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
