//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/converter.h"

#include "circt/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace dfcxx {

DFCIRConverter::DFCIRConverter(const DFLatencyConfig &config) {
  this->config = LatencyConfig();
  for (auto [op, latency]: config) {
    this->config[static_cast<mlir::dfcir::Ops>(op)] = latency;
  }
  this->config[mlir::dfcir::UNDEFINED] = 0;
}

bool DFCIRConverter::convertAndPrint(mlir::ModuleOp module,
                                     OutputStreams &outputStreams,
                                     const Scheduler &sched) {
  mlir::MLIRContext *context = module.getContext();
  mlir::PassManager pm(context);

  // Dump DFCIR if the corresponding option is specified.
  if (auto *stream = outputStreams[OUT_FORMAT_ID_INT(DFCIR)]) {
    module.print(*stream);
  }

  pm.addPass(mlir::dfcir::createDFCIRToFIRRTLPass(&config));
  switch (sched) {
    case Linear:
      pm.addPass(mlir::dfcir::createDFCIRLinearSchedulerPass());
      break;
    case ASAP:
      pm.addPass(mlir::dfcir::createDFCIRASAPSchedulerPass());
      break;
  }

  // Add FIRRTL->SystemVerilog passes if SystemVerilog output option is specified.
  if (auto *stream = outputStreams[OUT_FORMAT_ID_INT(SystemVerilog)]) {
    pm.addPass(circt::createLowerFIRRTLToHWPass());
    pm.addPass(circt::createLowerSeqToSVPass());
    pm.addPass(circt::createExportVerilogPass(*stream));
  }

  auto result = pm.run(module);
  return result.succeeded();
}

} // namespace dfcxx
