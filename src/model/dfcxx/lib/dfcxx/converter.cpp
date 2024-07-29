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
  pm.addPass(mlir::dfcir::createDFCIRToFIRRTLPass(&config));
  switch (sched) {
    case Linear:
      pm.addPass(mlir::dfcir::createDFCIRLinearSchedulerPass());
      break;
    case ASAP:
      pm.addPass(mlir::dfcir::createDFCIRASAPSchedulerPass());
      break;
  }
  pm.addPass(circt::createLowerFIRRTLToHWPass());
  pm.addPass(circt::createLowerSeqToSVPass());
  pm.addPass(circt::createExportVerilogPass(*(
      outputStreams[OUT_FORMAT_ID_INT(SystemVerilog)])));
  auto result = pm.run(module);
  return result.succeeded();
}

} // namespace dfcxx
