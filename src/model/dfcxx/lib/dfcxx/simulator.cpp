//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/simulator.h"
#include "dfcxx/types/types.h"
#include "dfcxx/vars/vars.h"

#include <string>

namespace dfcxx {

DFCXXSimulator::DFCXXSimulator(std::vector<Node> &nodes,
                               const Inputs &inputs) : nodes(nodes),
                                                       inputs(inputs) {}



uint64_t DFCXXSimulator::readInputData(std::ifstream &in,
                                       IOVars &inputMapping) {
  uint64_t ind = 0;
  std::string line;
  bool atLeastOne = false;
  while (std::getline(in, line) && ind < BUF_SIZE) {
    // An empty line is treated as a data block delimiter.
    if (line.empty()) {
      ++ind;
      continue;
    }
    size_t spaceInd = line.find(' ');
    if (spaceInd == line.npos ||
        spaceInd == 0 ||
        spaceInd == line.size() - 1) {
      return 0;
    }
    atLeastOne = true;
    inputMapping[line.substr(0, spaceInd)][ind] =
        std::stoul(line.substr(spaceInd + 1), 0, 16);
  }
  // It is assumed that at least one data block exists.
  return atLeastOne ? (ind + 1) : 0;
}

void DFCXXSimulator::processInput(RecordedValues &vals, Node &node,
                                  IOVars &input, uint64_t ind) {
  auto name = std::string(DFVariable(node.var).getName());
  vals[node] = input[name][ind];
}

void DFCXXSimulator::processOutput(RecordedValues &vals, Node &node,
                                   IOVars &output, uint64_t ind) {
  auto name = std::string(DFVariable(node.var).getName());
  // Take output's only connection and assign the existing source value.
  vals[node] = vals[inputs.at(node)[0].source];
  output[name][ind] = vals[node];
}

void DFCXXSimulator::processConst(RecordedValues &vals, Node &node) {
  vals[node] = ((DFConstant *) node.var)->getUInt();
}

void DFCXXSimulator::processMux(RecordedValues &vals, Node &node) {
  auto muxedValue = vals[inputs.at(node)[node.data.muxId].source];
  vals[node] = vals[inputs.at(node)[muxedValue + 1].source];
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::ADD>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left + right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] + vals[inputs.at(node)[1].source];
    }
  } else {
    double left = *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double right = *(reinterpret_cast<double*>(&vals[inputs.at(node)[1].source]));
    double result = left + right;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  }
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::SUB>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left - right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] - vals[inputs.at(node)[1].source];
    }
  } else {
    double left = *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double right = *(reinterpret_cast<double*>(&vals[inputs.at(node)[1].source]));
    double result = left - right;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  }
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::MUL>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left * right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] * vals[inputs.at(node)[1].source];
    }
  } else {
    double left = *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double right = *(reinterpret_cast<double*>(&vals[inputs.at(node)[1].source]));
    double result = left * right;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  }
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::DIV>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left / right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] / vals[inputs.at(node)[1].source];
    }
  } else {
    double left = *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double right = *(reinterpret_cast<double*>(&vals[inputs.at(node)[1].source]));
    double result = left / right;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  }
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::AND>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left & right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] & vals[inputs.at(node)[1].source];
    }
  }
  // No AND for floats.
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::OR>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left | right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] | vals[inputs.at(node)[1].source];
    }
  }
  // No OR for floats.
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::XOR>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left ^ right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] ^ vals[inputs.at(node)[1].source];
    }
  }
  // No XOR for floats.
}

template <>
void DFCXXSimulator::processUnaryOp<OpType::NOT>(RecordedValues &vals,
                                                 Node &node) {
  vals[node] = ~(vals[inputs.at(node)[0].source]);
}

template <>
void DFCXXSimulator::processUnaryOp<OpType::NEG>(RecordedValues &vals,
                                                 Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
    int64_t result = -left;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  } else {
    double left = *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double result = -left;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  }
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::LESS>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left < right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] < vals[inputs.at(node)[1].source];
    }
  } else {
    double left = *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double right = *(reinterpret_cast<double*>(&vals[inputs.at(node)[1].source]));
    double result = left < right;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  }
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::LESSEQ>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left <= right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] <= vals[inputs.at(node)[1].source];
    }
  } else {
    double left = *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double right = *(reinterpret_cast<double*>(&vals[inputs.at(node)[1].source]));
    double result = left <= right;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  }
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::GREATER>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left > right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] > vals[inputs.at(node)[1].source];
    }
  } else {
    double left = *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double right = *(reinterpret_cast<double*>(&vals[inputs.at(node)[1].source]));
    double result = left > right;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  }
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::GREATEREQ>(RecordedValues &vals,
                                                  Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t right = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));
      int64_t result = left >= right;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] >= vals[inputs.at(node)[1].source];
    }
  } else {
    double left = *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double right = *(reinterpret_cast<double*>(&vals[inputs.at(node)[1].source]));
    double result = left >= right;
    vals[node] = *(reinterpret_cast<SimValue*>(&result));
  }
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::EQ>(RecordedValues &vals,
                                                  Node &node) {
  vals[node] = vals[inputs.at(node)[0].source] == vals[inputs.at(node)[1].source];
}

template <>
void DFCXXSimulator::processBinaryOp<OpType::NEQ>(RecordedValues &vals,
                                                  Node &node) {
  vals[node] = vals[inputs.at(node)[0].source] != vals[inputs.at(node)[1].source];
}

void DFCXXSimulator::processShiftLeft(RecordedValues &vals, Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t result = left << node.data.bitShift;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] << node.data.bitShift;
    }
  }
  // No left bit shift for floats.
}

void DFCXXSimulator::processShiftRight(RecordedValues &vals, Node &node) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      int64_t left = *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
      int64_t result = left >> node.data.bitShift;
      vals[node] = *(reinterpret_cast<SimValue*>(&result));
    } else {
      vals[node] = vals[inputs.at(node)[0].source] >> node.data.bitShift;
    }
  }
  // No right bit shift for floats.
}

bool DFCXXSimulator::runSim(IOVars &input,
                            IOVars &output,
                            uint64_t count) {
  // Node->value mapping is initialized. This allows us
  // to rememeber the relevant value for the operand node.
  // With every single "clock" (loop iteration) input
  // nodes' mapping is updated with the value from the buffer.
  RecordedValues vals;
  for (uint64_t i = 0; i < count; ++i) {
    for (Node &node : nodes) {
      switch (node.type) {
        case OFFSET:
          return false;  // TODO: Add offset support in the future.
        case IN: processInput(vals, node, input, i); break;
        case OUT: processOutput(vals, node, output, i); break;
        case CONST: processConst(vals, node); break;
        case MUX: processMux(vals, node); break;
        case ADD: processBinaryOp<OpType::ADD>(vals, node); break;
        case SUB: processBinaryOp<OpType::SUB>(vals, node); break;
        case MUL: processBinaryOp<OpType::MUL>(vals, node); break;
        case DIV: processBinaryOp<OpType::DIV>(vals, node); break;
        case AND: processBinaryOp<OpType::AND>(vals, node); break;
        case OR: processBinaryOp<OpType::OR>(vals, node); break;
        case XOR: processBinaryOp<OpType::XOR>(vals, node); break;
        case NOT: processUnaryOp<OpType::NOT>(vals, node); break;
        case NEG: processUnaryOp<OpType::NEG>(vals, node); break;
        case LESS: processBinaryOp<OpType::LESS>(vals, node); break;
        case LESSEQ: processBinaryOp<OpType::LESSEQ>(vals, node); break;
        case GREATER: processBinaryOp<OpType::GREATER>(vals, node); break;
        case GREATEREQ: processBinaryOp<OpType::GREATEREQ>(vals, node); break;
        case EQ: processBinaryOp<OpType::EQ>(vals, node); break;
        case NEQ: processBinaryOp<OpType::NEQ>(vals, node); break;
        case SHL: processShiftLeft(vals, node); break;
        case SHR: processShiftRight(vals, node); break;
      }
    }
  }
  return true;
}

bool DFCXXSimulator::writeOutput(std::ofstream &out,
                                 IOVars &output,
                                 uint64_t count) {
  auto outFunc = [&out, &output] (uint64_t iter) {
    for (auto &kv : output) {
      out << kv.first << " 0x" << std::hex << kv.second[iter] << "\n";
    }
  };

  outFunc(0);
  for (uint64_t i = 1; i < count; ++i) {
    out << "\n";
    outFunc(i);
  }
  return true;
}

bool DFCXXSimulator::simulate(std::ifstream &in,
                              std::ofstream &out) {
  IOVars input;
  IOVars output;
  while (uint64_t count = readInputData(in, input)) {
    // If either the simulation itself or writing to output file
    // fails - return false.
    if (!runSim(input, output, count) ||
        !writeOutput(out, output, count)) {
      return false;
    }
  }
  return true;
}

} // namespace dfcxx
