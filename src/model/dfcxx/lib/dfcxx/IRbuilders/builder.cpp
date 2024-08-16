//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/IRbuilders/builder.h"
#include "dfcxx/utils.h"

#include "mlir/Parser/Parser.h"

namespace dfcxx {

DFCIRBuilder::DFCIRBuilder() : ctx(), builder(&ctx), conv(&ctx) {
  // We are allowed to initialize 'builder'-field before loading
  // dialects as OpBuilder only stores the pointer to MLIRContext
  // and doesn't check any of its state.
  ctx.getOrLoadDialect<mlir::dfcir::DFCIRDialect>();
}

void DFCIRBuilder::buildKernelBody(Graph *graph, mlir::OpBuilder &builder) {
  std::vector<Node> sorted = topSort(graph->startNodes,
                                     graph->outputs,
                                     graph->nodes.size());

  for (Node node : sorted) {
    translate(node, graph, builder);
  }
}

mlir::dfcir::KernelOp
DFCIRBuilder::buildKernel(dfcxx::Kernel *kern, mlir::OpBuilder &builder) {
  auto kernel = builder.create<mlir::dfcir::KernelOp>(builder.getUnknownLoc(),
                                                      kern->getName());
  builder.setInsertionPointToStart(&kernel.getBody().emplaceBlock());
  buildKernelBody(&kern->graph, builder);
  return kernel;
}

mlir::ModuleOp
DFCIRBuilder::buildModule(dfcxx::Kernel *kern, mlir::OpBuilder &builder) {
  auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  buildKernel(kern, builder);
  return module;
}

mlir::ModuleOp DFCIRBuilder::buildModule(dfcxx::Kernel *kern) {

  assert(ctx.getLoadedDialect<mlir::dfcir::DFCIRDialect>() != nullptr);

  mlir::OpBuilder builder(&ctx);
  module = buildModule(kern, builder);

  return module.get();
}

void DFCIRBuilder::translate(dfcxx::Node node, dfcxx::Graph *graph,
                             mlir::OpBuilder &builder) {
  auto loc = builder.getUnknownLoc();

  const auto &ins = graph->inputs[node];

  auto nameAttr = mlir::StringAttr::get(&ctx, node.var->getName());

  switch (node.type) {
    case OFFSET: {
      Node in = ins[0].source;
      auto type = mlir::IntegerType::get(builder.getContext(), 64,
                                         mlir::IntegerType::Signless);
      auto attr = mlir::IntegerAttr::get(type, node.data.offset);
      auto newOp = builder.create<mlir::dfcir::OffsetOp>(loc, conv[in.var],
                                                         map[in], attr);
      map[node] = newOp.getResult();
      break;
    }
    case IN: {
      if (node.var->isStream()) {
        auto newOp = builder.create<mlir::dfcir::InputOp>(loc, conv[node.var],
                                                          nameAttr, nullptr);
        map[node] = newOp.getResult();
      } else {
        auto newOp = builder.create<mlir::dfcir::ScalarInputOp>(loc,
                                                                conv[node.var],
                                                                nameAttr);
        map[node] = newOp.getResult();
      }
      break;
    }
    case OUT: {
      if (node.var->isStream()) {
        auto newOp = builder.create<mlir::dfcir::OutputOp>(loc, conv[node.var],
                                                           nameAttr, nullptr,
                                                           nullptr);
        map[node] = newOp.getResult();
      } else {
        auto newOp = builder.create<mlir::dfcir::ScalarOutputOp>(loc,
                                                                 conv[node.var],
                                                                 nameAttr,
                                                                 nullptr);
        map[node] = newOp.getResult();
      }
      break;
    }
    case CONST: {
      auto constant = (DFConstant *) (node.var);
      int64_t val;
      mlir::IntegerType attrType;
      unsigned width = constant->getType().getTotalBits();
      switch (constant->getKind()) {
        case INT:
          val = constant->getInt();
          attrType = mlir::IntegerType::get(builder.getContext(), width,
                                            mlir::IntegerType::Signed);
          break;
        case UINT: {
          auto tmpU = constant->getUInt();
          memcpy(&val, &tmpU, sizeof(val));
          attrType = mlir::IntegerType::get(builder.getContext(), width,
                                            mlir::IntegerType::Unsigned);
          break;
        }
        case FLOAT: {
          auto tmpD = constant->getDouble();
          memcpy(&val, &tmpD, sizeof(val));
          attrType = mlir::IntegerType::get(builder.getContext(), width,
                                            mlir::IntegerType::Signless);
          break;
        }
      }
      auto attr = mlir::IntegerAttr::get(attrType, val);
      auto newOp = builder.create<mlir::dfcir::ConstantOp>(loc, conv[node.var],
                                                           attr);
      map[node] = newOp.getRes();
      break;
    }
    case MUX: {
      Node ctrl = ins[node.data.muxId].source;
      llvm::SmallVector<mlir::Value> mux;
      uint64_t size = ins.size();
      for (uint64_t i = 0; i < size; ++i) {
        // To produce correct FIRRTL/SystemVerilog code
        // multiplexer inputs have to be reversed.
        uint64_t ind = size - 1 - i;
        if (ind != node.data.muxId) {
          mux.push_back(map[ins[ind].source]);
        }
      }
      muxMap[node] = mux;
      auto newOp = builder.create<mlir::dfcir::MuxOp>(loc, conv[node.var],
                                                      map[ctrl], muxMap[node]);
      map[node] = newOp.getRes();
      break;
    }
    case ADD: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::AddOp>(loc, conv[node.var],
                                                      map[first], map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case SUB: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::SubOp>(loc, conv[node.var],
                                                      map[first], map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case MUL: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::MulOp>(loc, conv[node.var],
                                                      map[first], map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case DIV: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::DivOp>(loc, conv[node.var],
                                                      map[first], map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case AND: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::AndOp>(loc, conv[node.var],
                                                      map[first], map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case OR: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::OrOp>(loc, conv[node.var],
                                                      map[first], map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case XOR: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::XorOp>(loc, conv[node.var],
                                                     map[first], map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case NOT: {
      Node first = ins[0].source;
      auto newOp = builder.create<mlir::dfcir::NotOp>(loc, conv[node.var],
                                                      map[first]);
      map[node] = newOp.getResult();
      break;
    }
    case NEG: {
      Node first = ins[0].source;
      auto newOp = builder.create<mlir::dfcir::NegOp>(loc, conv[node.var],
                                                      map[first]);
      map[node] = newOp.getResult();
      break;
    }
    case LESS: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::LessOp>(loc, conv[node.var],
                                                       map[first], map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case LESSEQ: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::LessEqOp>(loc, conv[node.var],
                                                         map[first],
                                                         map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case GREATER: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::GreaterOp>(loc, conv[node.var],
                                                          map[first],
                                                          map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case GREATEREQ: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::GreaterEqOp>(loc, conv[node.var],
                                                            map[first],
                                                            map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case EQ: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::EqOp>(loc, conv[node.var],
                                                     map[first],
                                                     map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case NEQ: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      auto newOp = builder.create<mlir::dfcir::NotEqOp>(loc, conv[node.var],
                                                        map[first],
                                                        map[second]);
      map[node] = newOp.getResult();
      break;
    }
    case SHL: {
      Node first = ins[0].source;
      auto attrType = mlir::IntegerType::get(builder.getContext(), 32,
                                             mlir::IntegerType::Signless);
      auto attr = mlir::IntegerAttr::get(attrType, node.data.bitShift);
                                  
      auto newOp = builder.create<mlir::dfcir::ShiftLeftOp>(loc, conv[node.var],
                                                            map[first], attr);
      map[node] = newOp.getResult();
      break;
    }
    case SHR: {
      Node first = ins[0].source;
      auto attrType = mlir::IntegerType::get(builder.getContext(), 32,
                                             mlir::IntegerType::Signless);
      auto attr = mlir::IntegerAttr::get(attrType, node.data.bitShift);
                                  
      auto newOp = builder.create<mlir::dfcir::ShiftRightOp>(loc, conv[node.var],
                                                             map[first], attr);
      map[node] = newOp.getResult();
      break;
    }
  }
  if (graph->connections.find(node) != graph->connections.end()) {
    auto conSrc = graph->connections.at(node).source;
    builder.create<mlir::dfcir::ConnectOp>(loc, map[node], map[conSrc]);
  }
}

} // namespace dfcxx
