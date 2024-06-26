//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/IRbuilders/builder.h"

namespace dfcxx {

std::stack<Node> DFCIRBuilder::topSortNodes(Graph *graph) {
  std::stack<Node> result;

  std::unordered_map<Node, size_t> checked;
  std::stack<Node> stack;

  for (Node node: graph->startNodes) {
    stack.push(node);
    checked[node] = 0;
  }

  while (!stack.empty()) {
    Node node = stack.top();
    size_t count = graph->outputs[node].size();
    size_t curr;
    bool flag = true;
    for (curr = checked[node]; flag && curr < count; ++curr) {
      Channel next = graph->outputs[node][curr];
      if (!checked[next.target]) {
        stack.push(next.target);
        flag = false;
      }
      ++checked[node];
    }

    if (flag) {
      stack.pop();
      result.push(node);
    }
  }
  return result;
}

void DFCIRBuilder::translate(dfcxx::Node node, dfcxx::Graph *graph,
                             mlir::OpBuilder &builder) {
  auto loc = builder.getUnknownLoc();

  const auto &ins = graph->inputs[node];
  const auto &outs = graph->outputs[node];

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
      for (int i = ins.size() - 1; i >= 0; --i) {
        if (i != node.data.muxId) {
          mux.push_back(map[ins[i].source]);
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