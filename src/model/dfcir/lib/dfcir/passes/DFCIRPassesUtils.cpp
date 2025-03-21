//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRPassesUtils.h"

#include <algorithm>
#include <cassert>
#include <stack>

namespace mlir::dfcir::utils {

void eraseOffsets(mlir::Operation *op) {
  op->walk([](OffsetOp offset) {
    auto input = offset->getOperand(0);
    auto result = offset->getResult(0);
    for (auto &operand: llvm::make_early_inc_range(result.getUses())) {
      operand.set(input);
    }
    offset->erase();
  });
}

Ops resolveInternalOpType(mlir::Operation *op) {
  auto resultType = op->getResult(0).getType();
  auto dfType = llvm::dyn_cast<DFType>(resultType).getDFType();
  bool isFloat = llvm::isa<DFCIRFloatType>(dfType);
 
  if (llvm::isa<AddOp>(op)) {
    return (isFloat) ? Ops::ADD_FLOAT : Ops::ADD_INT;
  } else if (llvm::isa<SubOp>(op)) {
    return (isFloat) ? Ops::SUB_FLOAT : Ops::SUB_INT;
  } else if (llvm::isa<MulOp>(op)) {
    return (isFloat) ? Ops::MUL_FLOAT : Ops::MUL_INT;
  } else if (llvm::isa<DivOp>(op)) {
    return (isFloat) ? Ops::DIV_FLOAT : Ops::DIV_INT;
  } else if (llvm::isa<NegOp>(op)) {
    return (isFloat) ? Ops::NEG_FLOAT : Ops::NEG_INT;
  } else if (llvm::isa<AndOp>(op)) {
    return (isFloat) ? Ops::AND_FLOAT : Ops::AND_INT;
  } else if (llvm::isa<OrOp>(op)) {
    return (isFloat) ? Ops::OR_FLOAT : Ops::OR_INT;
  } else if (llvm::isa<XorOp>(op)) {
    return (isFloat) ? Ops::XOR_FLOAT : Ops::XOR_INT;
  } else if (llvm::isa<NotOp>(op)) {
    return (isFloat) ? Ops::NOT_FLOAT : Ops::NOT_INT;
  } else if (llvm::isa<LessOp>(op)) {
    return (isFloat) ? Ops::LESS_FLOAT : Ops::LESS_INT;
  } else if (llvm::isa<LessEqOp>(op)) {
    return (isFloat) ? Ops::LESSEQ_FLOAT : Ops::LESSEQ_INT;
  } else if (llvm::isa<GreaterOp>(op)) {
    return (isFloat) ? Ops::GREATER_FLOAT : Ops::GREATER_INT;
  } else if (llvm::isa<GreaterEqOp>(op)) {
    return (isFloat) ? Ops::GREATEREQ_FLOAT : Ops::GREATEREQ_INT;
  } else if (llvm::isa<EqOp>(op)) {
    return (isFloat) ? Ops::EQ_FLOAT : Ops::EQ_INT;
  } else if (llvm::isa<NotEqOp>(op)) {
    return (isFloat) ? Ops::NEQ_FLOAT : Ops::NEQ_INT;
  }

  return Ops::UNDEFINED;
}

std::string opTypeToString(const Ops &opType) {
  switch (opType) {
  case ADD_INT: return "ADD_INT";
  case ADD_FLOAT: return "ADD_FLOAT";
  case SUB_INT: return "SUB_INT";
  case SUB_FLOAT: return "SUB_FLOAT";
  case MUL_INT: return "MUL_INT";
  case MUL_FLOAT: return "MUL_FLOAT";
  case DIV_INT: return "DIV_INT";
  case DIV_FLOAT: return "DIV_FLOAT";
  case NEG_INT: return "NEG_INT";
  case NEG_FLOAT: return "NEG_FLOAT";
  case AND_INT: return "AND_INT";
  case AND_FLOAT: return "AND_FLOAT";
  case OR_INT: return "OR_INT";
  case OR_FLOAT: return "OR_FLOAT";
  case XOR_INT: return "XOR_INT";
  case XOR_FLOAT: return "XOR_FLOAT";
  case NOT_INT: return "NOT_INT";
  case NOT_FLOAT: return "NOT_FLOAT";
  case LESS_INT: return "LESS_INT";
  case LESS_FLOAT: return "LESS_FLOAT";
  case LESSEQ_INT: return "LESSEQ_INT";
  case LESSEQ_FLOAT: return "LESSEQ_FLOAT";
  case GREATER_INT: return "GREATER_INT";
  case GREATER_FLOAT: return "GREATER_FLOAT";
  case GREATEREQ_INT: return "GREATEREQ_INT";
  case GREATEREQ_FLOAT: return "GREATEREQ_FLOAT";
  case EQ_INT: return "EQ_INT";
  case EQ_FLOAT: return "EQ_FLOAT";
  case NEQ_INT: return "NEQ_INT";
  case NEQ_FLOAT: return "NEQ_FLOAT";
  default: return "<unknown>";
  }
}

} // namespace mlir::dfcir::utils
