//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/conversions/DFCIRPassesUtils.h"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "ctemplate/template.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <string>

namespace mlir::dfcir {

#define GEN_PASS_DECL_FIRRTLSTUBGENERATORPASS
#define GEN_PASS_DEF_FIRRTLSTUBGENERATORPASS

#include "dfcir/conversions/DFCIRPasses.h.inc"

class FIRRTLStubGeneratorPass
    : public impl::FIRRTLStubGeneratorPassBase<FIRRTLStubGeneratorPass> {
  using TemplateDictionary = ctemplate::TemplateDictionary;
  using FExtModuleOp = circt::firrtl::FExtModuleOp;
  using CircuitOp = circt::firrtl::CircuitOp;
  using FIRRTLBaseType = circt::firrtl::FIRRTLBaseType;
  using IntType = circt::firrtl::IntType;

private:
  TemplateDictionary *processFIFOModule(TemplateDictionary *dict,
                                        FExtModuleOp module) {
    TemplateDictionary *result = dict->AddSectionDictionary("FIFO_MODULES");
    auto ports = module.getPorts();
    auto res1Type = llvm::cast<FIRRTLBaseType>(module.getPortType(0));
    int32_t width = res1Type.getBitWidthOrSentinel();
    result->SetFormattedValue("WIDTH", "%d", width - 1);
    result->SetValue("RES1", ports[0].getName().data());
    result->SetValue("ARG1", ports[1].getName().data());
    result->SetValue("CLK", ports[2].getName().data());
    return result;
  }

  TemplateDictionary *processBinModule(TemplateDictionary *dict,
                                       FExtModuleOp module,
                                       unsigned latency) {
    TemplateDictionary *result = dict->AddSectionDictionary("BINARY_MODULES");
    auto moduleName = module.getModuleName();
    if (moduleName.contains(ADD_MODULE)) {
      result->SetValue("OP", "+");
    } else if (moduleName.contains(SUB_MODULE)) {
      result->SetValue("OP", "-");
    } else {
      result->SetValue("OP", "*");
    }
    auto ports = module.getPorts();
    auto res1Type = llvm::cast<FIRRTLBaseType>(module.getPortType(0));
    bool isSigned = llvm::cast<IntType>(res1Type).isSigned();
    int32_t width3 = res1Type.getBitWidthOrSentinel();
    result->SetFormattedValue("WIDTH3", "%d", width3 - 1);
    auto arg1Type = llvm::cast<FIRRTLBaseType>(module.getPortType(1));
    int32_t width1 = arg1Type.getBitWidthOrSentinel();
    result->SetFormattedValue("WIDTH1", "%d", width1 - 1);
    auto arg2Type = llvm::cast<FIRRTLBaseType>(module.getPortType(2));
    int32_t width2 = arg2Type.getBitWidthOrSentinel();
    result->SetFormattedValue("WIDTH2", "%d", width2 - 1);
    int32_t rWidth = std::max(width1, width2);
    result->SetFormattedValue("RWIDTH", "%d", rWidth - 1);
    result->SetValue("RES1", ports[0].getName().data());
    auto arg1 = ports[1].getName().data();
    result->SetValue("ARG1", arg1);
    auto arg2 = ports[2].getName().data();
    result->SetValue("ARG2", arg2);
    result->SetValue("CLK", ports[3].getName().data());
    int32_t repeat1 = std::max(rWidth - width1, 0);
    result->SetFormattedValue("REPEAT1", "%d", repeat1);
    int32_t repeat2 = std::max(rWidth - width2, 0);
    result->SetFormattedValue("REPEAT2", "%d", repeat2);
    int32_t repeat3 = std::max(width3 - rWidth, 0);
    result->SetFormattedValue("REPEAT3", "%d", repeat3);
    int32_t diff = std::min(width3, rWidth);
    result->SetFormattedValue("DIFF", "%d", diff - 1);
    int32_t cat = std::max(width3, rWidth);
    result->SetFormattedValue("CAT", "%d", cat - 1);
    if (isSigned) {
        result->SetFormattedValue("REPEAT_VAL1", "%s[%d]", arg1, width1 - 1);
        result->SetFormattedValue("REPEAT_VAL2", "%s[%d]", arg2, width2 - 1);
        result->SetFormattedValue("REPEAT_VAL3", "r[%d][%d]",
                                  latency - 1, rWidth - 1);
    } else {
        result->SetValue("REPEAT_VAL1", "1'h0");
        result->SetValue("REPEAT_VAL2", "1'h0");
        result->SetValue("REPEAT_VAL3", "1'h0");
    }
    return result;
  }

  void fillDictionary(TemplateDictionary *dict, CircuitOp circuit) {
    Block *block = circuit.getBodyBlock();
    auto begin = block->op_begin<FExtModuleOp>();
    auto end = block->op_end<FExtModuleOp>();
    for (auto op = begin; op != end; ++op) {
      auto moduleName = (*op).getModuleName();
      unsigned latency =
          (*op)->getAttr(INSTANCE_LATENCY_ATTR)
              .cast<IntegerAttr>().getUInt();
      TemplateDictionary *moduleDict =
          moduleName.contains(BUF_MODULE) ?
          processFIFOModule(dict, *op) :
          processBinModule(dict, *op, latency);
      moduleDict->SetFormattedValue("LATENCY", "%u", latency - 1);
      moduleDict->SetValue("MODULE_NAME", moduleName.data());
    }
  }

  std::string generateOutput() {
    std::string result;
    TemplateDictionary *topLevelDict = new TemplateDictionary("stubs");

    mlir::Operation *op = getOperation();
    CircuitOp circuit = mlir::utils::findFirstOccurence<CircuitOp>(op);

    fillDictionary(topLevelDict, circuit);
    ctemplate::ExpandTemplate(STUBS_TEMPLATE_PATH, ctemplate::DO_NOT_STRIP,
                              topLevelDict, &result);
    return result;
  }

public:
  explicit FIRRTLStubGeneratorPass(const FIRRTLStubGeneratorPassOptions &opt)
      : impl::FIRRTLStubGeneratorPassBase<FIRRTLStubGeneratorPass>(opt) {}

  void runOnOperation() override {
    const std::string &output = generateOutput();

    *stream << output;
  }
};

std::unique_ptr<mlir::Pass>
    createFIRRTLStubGeneratorPass(llvm::raw_ostream *stream) {
  FIRRTLStubGeneratorPassOptions options;
  options.stream = stream;
  return std::make_unique<FIRRTLStubGeneratorPass>(options);
}

} // namespace mlir::dfcir
