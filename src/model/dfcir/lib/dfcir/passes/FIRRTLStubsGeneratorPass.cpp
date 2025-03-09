//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/passes/DFCIRPassesUtils.h"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "ctemplate/template.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace mlir::dfcir {

#define GEN_PASS_DECL_FIRRTLSTUBGENERATORPASS
#define GEN_PASS_DEF_FIRRTLSTUBGENERATORPASS
#include "dfcir/passes/DFCIRPasses.h.inc"

#include "dfcir/passes/ModuleDefines.inc"

class FIRRTLStubGeneratorPass
    : public impl::FIRRTLStubGeneratorPassBase<FIRRTLStubGeneratorPass> {
  using TemplateDictionary = ctemplate::TemplateDictionary;
  using FExtModuleOp = circt::firrtl::FExtModuleOp;
  using CircuitOp = circt::firrtl::CircuitOp;
  using FIRRTLBaseType = circt::firrtl::FIRRTLBaseType;
  using IntType = circt::firrtl::IntType;

private:
  TemplateDictionary *processFIFOModule(TemplateDictionary *dict,
                                        FExtModuleOp module,
                                        uint32_t latency) {
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

  TemplateDictionary *processSimpleCastModule(TemplateDictionary *dict,
                                              FExtModuleOp module,
                                              uint32_t latency) {
    std::string_view templateName =
        (latency) ? "CAST_MODULES" : "COMB_CAST_MODULES";
    TemplateDictionary *result = dict->AddSectionDictionary(templateName.data());

    auto ports = module.getPorts();
    result->SetValue("RES1", ports[0].getName().data());
    auto arg1 = ports[1].getName().data();
    result->SetValue("ARG1", arg1);
    result->SetValue("CLK", ports[2].getName().data());

    auto res1Type = llvm::cast<FIRRTLBaseType>(module.getPortType(0));
    bool isSigned = llvm::cast<IntType>(res1Type).isSigned();
    int32_t width2 = res1Type.getBitWidthOrSentinel();
    result->SetFormattedValue("WIDTH2", "%d", width2 - 1);

    auto arg1Type = llvm::cast<FIRRTLBaseType>(module.getPortType(1));
    int32_t width1 = arg1Type.getBitWidthOrSentinel();
    result->SetFormattedValue("WIDTH1", "%d", width1 - 1);

    int32_t repeat = std::max(width2 - width1, 0);
    result->SetFormattedValue("REPEAT", "%d", repeat);

    int32_t cat = std::max(width2, width1);
    result->SetFormattedValue("CAT", "%d", cat - 1);

    if (isSigned) {
      result->SetFormattedValue("REPEAT_VAL", "%s[%d]", arg1, width1 - 1);
    } else {
      result->SetValue("REPEAT_VAL", "1'h0");
    }

    return result;
  }

  TemplateDictionary *processBinModule(TemplateDictionary *dict,
                                       FExtModuleOp module,
                                       uint32_t latency) {
    std::string_view opChar;
    auto moduleName = module.getModuleName();
    bool isCompOp = false;
    if (moduleName.contains(ADD_MODULE)) {
      opChar = "+";
    } else if (moduleName.contains(SUB_MODULE)) {
      opChar = "-";
    } else if (moduleName.contains(MUL_MODULE)) {
      opChar = "*";
    } else if (moduleName.contains(LESSEQ_MODULE)) {
      opChar = "<="; isCompOp = true;
    } else if (moduleName.contains(GREATEREQ_MODULE)) {
      opChar = ">="; isCompOp = true;
    } else if (moduleName.contains(LESS_MODULE)) {
      opChar = "<"; isCompOp = true;
    }  else if (moduleName.contains(GREATER_MODULE)) {
      opChar = ">"; isCompOp = true;
    } else if (moduleName.contains(EQ_MODULE)) {
      opChar = "=="; isCompOp = true;
    } else if (moduleName.contains(NEQ_MODULE)) {
      opChar = "!="; isCompOp = true;
    } else if (moduleName.contains(AND_MODULE)) {
      opChar = "&";
    } else if (moduleName.contains(OR_MODULE)) {
      opChar = "|";
    } else if (moduleName.contains(XOR_MODULE)) {
      opChar = "^";
    } else {
      std::cout << "Unsupported binary operation:" << std::endl;
      module.dump();
      return nullptr;
    }

    std::string_view templateName =
        (latency) ? "BINARY_MODULES" : "COMB_BINARY_MODULES";
    TemplateDictionary *result = dict->AddSectionDictionary(templateName.data());

    result->SetValue("OP", opChar.data());
    auto ports = module.getPorts();
    result->SetValue("RES1", ports[0].getName().data());
    auto arg1 = ports[1].getName().data();
    result->SetValue("ARG1", arg1);
    auto arg2 = ports[2].getName().data();
    result->SetValue("ARG2", arg2);
    result->SetValue("CLK", ports[3].getName().data());

    auto res1Type = llvm::cast<FIRRTLBaseType>(module.getPortType(0));
    bool isSigned3 = llvm::cast<IntType>(res1Type).isSigned();
    std::string_view signed3 = (isSigned3) ? "signed " : "";
    result->SetFormattedValue("SIGNED3", "%s", signed3.data());
    int32_t width3 = res1Type.getBitWidthOrSentinel();
    result->SetFormattedValue("WIDTH3", "%d", width3 - 1);

    auto arg1Type = llvm::cast<FIRRTLBaseType>(module.getPortType(1));
    bool isSigned1 = llvm::cast<IntType>(arg1Type).isSigned();
    std::string_view signed1 = (isSigned1) ? "signed " : "";
    result->SetFormattedValue("SIGNED1", "%s", signed1.data());
    int32_t width1 = arg1Type.getBitWidthOrSentinel();
    result->SetFormattedValue("WIDTH1", "%d", width1 - 1);

    auto arg2Type = llvm::cast<FIRRTLBaseType>(module.getPortType(2));
    bool isSigned2 = llvm::cast<IntType>(arg2Type).isSigned();
    std::string_view signed2 = (isSigned2) ? "signed " : "";
    result->SetFormattedValue("SIGNED2", "%s", signed2.data());
    int32_t width2 = arg2Type.getBitWidthOrSentinel();
    result->SetFormattedValue("WIDTH2", "%d", width2 - 1);

    int32_t rWidth =
        (isCompOp) ? 1 : std::max(width1, width2);
    result->SetFormattedValue("RWIDTH", "%d", rWidth - 1);

    int32_t preparedWidth = std::max(width1, width2);
    result->SetFormattedValue("PREPARED_WIDTH", "%d", preparedWidth - 1);

    int32_t repeat1 = std::max(rWidth - width1, 0);
    result->SetFormattedValue("REPEAT1", "%d", repeat1);
    int32_t repeat2 = std::max(rWidth - width2, 0);
    result->SetFormattedValue("REPEAT2", "%d", repeat2);
    int32_t repeat3 = std::max(width3 - rWidth, 0);
    result->SetFormattedValue("REPEAT3", "%d", repeat3);

    int32_t cat = std::max(width3, rWidth);
    result->SetFormattedValue("CAT", "%d", cat - 1);

    if (isSigned3) {
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

  LogicalResult fillDictionary(TemplateDictionary *dict, CircuitOp circuit) {
    Block *block = circuit.getBodyBlock();
    auto begin = block->op_begin<FExtModuleOp>();
    auto end = block->op_end<FExtModuleOp>();
    for (auto op = begin; op != end; ++op) {
      auto moduleName = (*op).getModuleName();
      uint32_t latency =
          (*op)->getAttr(INSTANCE_LATENCY_ATTR)
              .cast<IntegerAttr>().getUInt();
      TemplateDictionary *moduleDict;
      if (moduleName.contains(BUF_MODULE)) {
        moduleDict = processFIFOModule(dict, *op, latency);
      } else if (moduleName.contains(CAST_MODULE)) {
        moduleDict = processSimpleCastModule(dict, *op, latency);
      } else {
        moduleDict = processBinModule(dict, *op, latency);
      }

      if (!moduleDict) {
        continue;
      }

      // It is assumed that for "latency" == 0 respective
      // "process*Module" methods know how to handle it,
      // so here we can subtract 1 from "latency"
      // without handling the mentioned case.
      moduleDict->SetFormattedValue("LATENCY", "%u", latency - 1);
      moduleDict->SetValue("MODULE_NAME", moduleName.data());
    }
    auto time = std::time(nullptr);
    auto *localTime = std::localtime(&time);
    dict->SetFormattedValue("GEN_TIME",
                            "%d-%d-%d %d:%d:%d",
                            localTime->tm_mday,
                            localTime->tm_mon + 1,
                            localTime->tm_year + 1900,
                            localTime->tm_hour,
                            localTime->tm_min,
                            localTime->tm_sec);
    return success();
  }

  std::optional<std::string> generateOutput() {
    std::string result;
    TemplateDictionary *topLevelDict = new TemplateDictionary("stubs");

    mlir::Operation *op = getOperation();
    CircuitOp circuit = mlir::utils::findFirstOccurence<CircuitOp>(op);

    if (failed(fillDictionary(topLevelDict, circuit))) {
      delete topLevelDict;
      return {};
    }
    ctemplate::ExpandTemplate(STUBS_TEMPLATE_PATH, ctemplate::DO_NOT_STRIP,
                              topLevelDict, &result);
    delete topLevelDict;
    return result;
  }

public:
  explicit FIRRTLStubGeneratorPass(const FIRRTLStubGeneratorPassOptions &opt)
      : impl::FIRRTLStubGeneratorPassBase<FIRRTLStubGeneratorPass>(opt) {}

  void runOnOperation() override {
    auto outputOrError = generateOutput();
    
    if (!outputOrError) {
      return signalPassFailure();
    }

    *stream << *outputOrError;
  }
};

std::unique_ptr<mlir::Pass>
    createFIRRTLStubGeneratorPass(llvm::raw_ostream *stream) {
  FIRRTLStubGeneratorPassOptions options;
  options.stream = stream;
  return std::make_unique<FIRRTLStubGeneratorPass>(options);
}

} // namespace mlir::dfcir
