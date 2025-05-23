//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/dfcir_processor.h"

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <memory>

namespace dfcxx {

class DFCIRDumperPass: public mlir::PassWrapper<DFCIRDumperPass, mlir::OperationPass<mlir::ModuleOp>> {
private:
  llvm::raw_fd_ostream *stream;

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DFCIRDumperPass)

  DFCIRDumperPass() = default;

  DFCIRDumperPass(llvm::raw_fd_ostream *stream) : DFCIRDumperPass() {
    this->stream = stream;
  }

  void runOnOperation() override {
    return getOperation()->print(*stream);
  }

};

std::unique_ptr<mlir::Pass> createDFCIRDumperPass(llvm::raw_fd_ostream *stream) {
  return std::make_unique<DFCIRDumperPass>(stream);
}

DFCIRProcessor::DFCIRProcessor(const DFLatencyConfig &config) {
  for (auto [op, latency]: config.internalOps) {
    this->config.internalOps[static_cast<mlir::dfcir::Ops>(op)] = latency;
  }
  this->config.internalOps[mlir::dfcir::UNDEFINED] = 0;
  this->config.externalOps = config.externalOps;
}

bool DFCIRProcessor::convertAndPrint(mlir::ModuleOp module,
                                     OutputStreams &outputStreams,
                                     const DFOptionsConfig &options) {
  mlir::MLIRContext *context = module.getContext();
  context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>();
  context->getOrLoadDialect<circt::sv::SVDialect>();
  mlir::PassManager pm(context);

  // Dump unscheduled DFCIR if the corresponding option is specified.
  if (auto *stream = outputStreams[OUT_FORMAT_ID_INT(UnscheduledDFCIR)]) {
    module.print(*stream);
  }

  switch (options.scheduler) {
    case Linear:
      pm.addPass(mlir::dfcir::createDFCIRLinearSchedulerPass(&config));
      break;
    case ASAP:
      pm.addPass(mlir::dfcir::createDFCIRASAPSchedulerPass(&config));
      break;
    case CombPipelining:
      pm.addPass(mlir::dfcir::createDFCIRCombPipelinePassPass(options.stages));
      break;
  }

  // Dump scheduled DFCIR if the corresponding option is specified.
  if (auto *stream = outputStreams[OUT_FORMAT_ID_INT(ScheduledDFCIR)]) {
    pm.addPass(createDFCIRDumperPass(stream));
  }

  pm.addPass(mlir::dfcir::createDFCIRToFIRRTLPass());

  // Dump FIRRTL if the corresponding option is specified.
  if (auto *stream = outputStreams[OUT_FORMAT_ID_INT(FIRRTL)]) {
    pm.addPass(createDFCIRDumperPass(stream));
  }

  // Add SystemVerilog library generation pass if the corresponding option
  // is specified.
  if (auto *stream = outputStreams[OUT_FORMAT_ID_INT(SVLibrary)]) {
    pm.addPass(mlir::dfcir::createFIRRTLStubGeneratorPass(stream));
  }

  // Add FIRRTL->SystemVerilog passes if SystemVerilog output
  // option is specified.
  if (auto *stream = outputStreams[OUT_FORMAT_ID_INT(SystemVerilog)]) {
    pm.addPass(circt::firrtl::createInnerSymbolDCEPass());

    mlir::OpPassManager &nestedCirct = pm.nest<circt::firrtl::CircuitOp>();
    nestedCirct.addPass(circt::firrtl::createIMConstPropPass());

    mlir::OpPassManager &nestedFirrtl = pm.nest<circt::firrtl::FModuleOp>();
    nestedFirrtl.addPass(circt::firrtl::createEliminateWiresPass());

    pm.addPass(circt::firrtl::createIMDeadCodeElimPass());
    pm.addPass(circt::createLowerFIRRTLToHWPass());

    mlir::OpPassManager &nestedHw = pm.nest<circt::hw::HWModuleOp>();
    nestedHw.addPass(circt::sv::createHWCleanupPass());
    nestedHw.addPass(mlir::createCSEPass());
    nestedHw.addPass(circt::sv::createPrettifyVerilogPass());

    pm.addPass(circt::createLowerSeqToSVPass());

    pm.addPass(circt::createExportVerilogPass(*stream));
  }

  auto result = pm.run(module);
  return result.succeeded();
}

} // namespace dfcxx
