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

DFCIRBuilder::DFCIRBuilder() : ctx(), conv(&ctx) {
  ctx.getOrLoadDialect<mlir::dfcir::DFCIRDialect>();
}

void DFCIRBuilder::buildKernelBody(const Graph &graph, mlir::OpBuilder &builder) {
  std::vector<Node> sorted = topSort(graph);

  std::unordered_map<Node, mlir::Value> map;
  for (Node node : sorted) {
    translate(node, graph, builder, map);
  }
}

mlir::dfcir::KernelOp DFCIRBuilder::buildKernel(Kernel *kern,
                                                mlir::OpBuilder &builder) {
  auto kernel = builder.create<mlir::dfcir::KernelOp>(builder.getUnknownLoc(),
                                                      kern->getName());
  builder.setInsertionPointToStart(&kernel.getBody().emplaceBlock());
  buildKernelBody(kern->getGraph(), builder);
  return kernel;
}

mlir::ModuleOp DFCIRBuilder::buildModule(Kernel *kern) {

  assert(ctx.getLoadedDialect<mlir::dfcir::DFCIRDialect>() != nullptr);

  mlir::OpBuilder builder(&ctx);
  module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module->getBody());
  buildKernel(kern, builder);

  return module.get();
}

#define MLIR_INT_ATTR_WIDTH 64

void DFCIRBuilder::translate(Node node, const Graph &graph,
                             mlir::OpBuilder &builder,
                             std::unordered_map<Node, mlir::Value> &map) {
  auto loc = builder.getUnknownLoc();
  
  const auto &ins = graph.getInputs().at(node);

  auto nameAttr = mlir::StringAttr::get(&ctx, node.var->getName());

  mlir::Operation *newOp = nullptr;

  switch (node.type) {
    case OFFSET: {
      Node in = ins[0].source;
      auto type = mlir::IntegerType::get(builder.getContext(),
                                         MLIR_INT_ATTR_WIDTH,
                                         mlir::IntegerType::Signless);
      auto attr = mlir::IntegerAttr::get(type, node.data.offset);
      newOp = builder.create<mlir::dfcir::OffsetOp>(loc, conv[in.var],
                                                    map[in], attr);
      break;
    }
    case IN: {
      if (node.var->isStream()) {
        newOp = builder.create<mlir::dfcir::InputOp>(loc, conv[node.var],
                                                     nameAttr, nullptr);
      } else {
        newOp = builder.create<mlir::dfcir::ScalarInputOp>(loc,
                                                           conv[node.var],
                                                           nameAttr);
      }
      break;
    }
    case OUT: {
      if (node.var->isStream()) {
        newOp = builder.create<mlir::dfcir::OutputOp>(loc, conv[node.var],
                                                      nameAttr, nullptr,
                                                      nullptr);
      } else {
        newOp = builder.create<mlir::dfcir::ScalarOutputOp>(loc,
                                                            conv[node.var],
                                                            nameAttr,
                                                            nullptr);
      }
      break;
    }
    case CONST: {
      auto constant = (DFConstant *) (node.var);
      int64_t val;
      mlir::IntegerType attrType;
      unsigned width = constant->getType()->getTotalBits();
      switch (constant->getKind()) {
        case DFConstant::TypeKind::INT:
          val = constant->getInt();
          attrType = mlir::IntegerType::get(builder.getContext(), width,
                                            mlir::IntegerType::Signed);
          break;
        case DFConstant::TypeKind::UINT: {
          auto tmpU = constant->getUInt();
          memcpy(&val, &tmpU, sizeof(val));
          attrType = mlir::IntegerType::get(builder.getContext(), width,
                                            mlir::IntegerType::Unsigned);
          break;
        }
        case DFConstant::TypeKind::FLOAT: {
          auto tmpD = constant->getDouble();
          memcpy(&val, &tmpD, sizeof(val));
          attrType = mlir::IntegerType::get(builder.getContext(), width,
                                            mlir::IntegerType::Signless);
          break;
        }
      }
      auto attr = mlir::IntegerAttr::get(attrType, val);
      newOp = builder.create<mlir::dfcir::ConstantOp>(loc, conv[node.var],
                                                      attr);
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

      newOp = builder.create<mlir::dfcir::MuxOp>(loc, conv[node.var],
                                                 map[ctrl], mux);
      break;
    }
    case ADD: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::AddOp>(loc, conv[node.var],
                                                 map[first], map[second]);
      break;
    }
    case SUB: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::SubOp>(loc, conv[node.var],
                                                 map[first], map[second]);
      break;
    }
    case MUL: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::MulOp>(loc, conv[node.var],
                                                 map[first], map[second]);
      break;
    }
    case DIV: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::DivOp>(loc, conv[node.var],
                                                 map[first], map[second]);
      break;
    }
    case AND: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::AndOp>(loc, conv[node.var],
                                                 map[first], map[second]);
      break;
    }
    case OR: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::OrOp>(loc, conv[node.var],
                                                map[first], map[second]);
      break;
    }
    case XOR: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::XorOp>(loc, conv[node.var],
                                                 map[first], map[second]);
      break;
    }
    case NOT: {
      Node first = ins[0].source;
      newOp = builder.create<mlir::dfcir::NotOp>(loc, conv[node.var],
                                                 map[first]);
      break;
    }
    case NEG: {
      Node first = ins[0].source;
      newOp = builder.create<mlir::dfcir::NegOp>(loc, conv[node.var],
                                                 map[first]);
      break;
    }
    case LESS: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::LessOp>(loc, conv[node.var],
                                                  map[first], map[second]);
      break;
    }
    case LESSEQ: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::LessEqOp>(loc, conv[node.var],
                                                    map[first],
                                                    map[second]);
      break;
    }
    case GREATER: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::GreaterOp>(loc, conv[node.var],
                                                     map[first],
                                                     map[second]);
      break;
    }
    case GREATEREQ: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::GreaterEqOp>(loc, conv[node.var],
                                                       map[first],
                                                       map[second]);
      break;
    }
    case EQ: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::EqOp>(loc, conv[node.var],
                                                map[first],
                                                map[second]);
      break;
    }
    case NEQ: {
      Node first = ins[0].source;
      Node second = ins[1].source;
      newOp = builder.create<mlir::dfcir::NotEqOp>(loc, conv[node.var],
                                                   map[first],
                                                   map[second]);
      break;
    }
    case SHL: {
      Node first = ins[0].source;
      auto attrType = mlir::IntegerType::get(builder.getContext(), 32,
                                             mlir::IntegerType::Signless);
      auto attr = mlir::IntegerAttr::get(attrType, node.data.bitShift);
                                  
      newOp = builder.create<mlir::dfcir::ShiftLeftOp>(loc, conv[node.var],
                                                       map[first], attr);
      break;
    }
    case SHR: {
      Node first = ins[0].source;
      auto attrType = mlir::IntegerType::get(builder.getContext(), 32,
                                             mlir::IntegerType::Signless);
      auto attr = mlir::IntegerAttr::get(attrType, node.data.bitShift);
                                  
      newOp = builder.create<mlir::dfcir::ShiftRightOp>(loc, conv[node.var],
                                                        map[first], attr);
      break;
    }
    default: assert(false);
  }

  map[node] = newOp->getResult(0);

  auto &connections = graph.getConnections();
  if (connections.find(node) != connections.end()) {
    auto conSrc = connections.at(node).source;
    builder.create<mlir::dfcir::ConnectOp>(loc, map[node], map[conSrc]);
  }
}

} // namespace dfcxx
