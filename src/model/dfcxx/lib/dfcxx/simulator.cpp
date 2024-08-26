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

uint64_t DFCXXSimulator::readInput(std::ifstream &in,
                                   IOVars &inData) {
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
    inData[line.substr(0, spaceInd)][ind] =
        std::stoul(line.substr(spaceInd + 1), 0, 16);
  }
  // It is assumed that at least one data block exists.
  return atLeastOne ? (ind + 1) : 0;
}

static bool processInput(RecordedValues &vals, const Node &node,
                         const Inputs &inputs, const IOVars &inData,
                         IOVars &outData, uint64_t ind) {
  auto name = std::string(DFVariable(node.var).getName());
  vals[node] = inData.at(name)[ind];
  return true;
}

static bool processOutput(RecordedValues &vals, const Node &node,
                          const Inputs &inputs, const IOVars &inData,
                          IOVars &outData, uint64_t ind) {
  auto name = std::string(DFVariable(node.var).getName());
  // Take output's only connection and assign the existing source value.
  vals[node] = vals[inputs.at(node)[0].source];
  outData[name][ind] = vals[node];
  return true;
}

static bool processConst(RecordedValues &vals, const Node &node,
                         const Inputs &inputs, const IOVars &inData,
                         IOVars &outData, uint64_t ind) {
  vals[node] = ((DFConstant *) node.var)->getUInt();
  return true;
}

static bool processMux(RecordedValues &vals, const Node &node,
                       const Inputs &inputs, const IOVars &inData,
                       IOVars &outData, uint64_t ind) {
  auto muxedValue = vals[inputs.at(node)[node.data.muxId].source];
  vals[node] = vals[inputs.at(node)[muxedValue + 1].source];
  return true;
}

#define GENERIC_FUNC_NAME(OP_NAME) process##OP_NAME##Name
#define PROCESS_GENERIC_BINARY_OP_FUNC(OP_NAME, OP)                           \
static bool GENERIC_FUNC_NAME(OP_NAME)(RecordedValues &vals,                  \
                                       const Node &node,                      \
                                       const Inputs &inputs,                  \
                                       const IOVars &inData,                  \
                                       IOVars &outData,                       \
                                       uint64_t ind) {                        \
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();              \
  if (type->isFixed()) {                                                      \
    if (((FixedType*) type)->isSigned()) {                                    \
      int64_t left =                                                          \
          *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));    \
      int64_t right =                                                         \
          *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[1].source]));    \
      int64_t result = left OP right;                                         \
      vals[node] =                                                            \
          *(reinterpret_cast<SimValue*>(&result));                            \
    } else {                                                                  \
      vals[node] =                                                            \
          vals[inputs.at(node)[0].source] OP vals[inputs.at(node)[1].source]; \
    }                                                                         \
  } else if (type->isFloat()) {                                               \
    double left =                                                             \
        *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));       \
    double right =                                                            \
        *(reinterpret_cast<double*>(&vals[inputs.at(node)[1].source]));       \
    double result = left OP right;                                            \
    vals[node] =                                                              \
        *(reinterpret_cast<SimValue*>(&result));                              \
  } else {                                                                    \
    return false;                                                             \
  }                                                                           \
  return true;                                                                \
}

PROCESS_GENERIC_BINARY_OP_FUNC(Add, +)

PROCESS_GENERIC_BINARY_OP_FUNC(Sub, -)

PROCESS_GENERIC_BINARY_OP_FUNC(Mul, *)

PROCESS_GENERIC_BINARY_OP_FUNC(Div, /)

PROCESS_GENERIC_BINARY_OP_FUNC(Less, <)

PROCESS_GENERIC_BINARY_OP_FUNC(LessEq, <=)

PROCESS_GENERIC_BINARY_OP_FUNC(Greater, >)

PROCESS_GENERIC_BINARY_OP_FUNC(GreaterEq, >=)

PROCESS_GENERIC_BINARY_OP_FUNC(Eq, ==)

PROCESS_GENERIC_BINARY_OP_FUNC(Neq, !=)

#define PROCESS_GENERIC_BITWISE_BINARY_OP_FUNC(OP_NAME, OP)               \
static bool GENERIC_FUNC_NAME(OP_NAME)(RecordedValues &vals,              \
                                       const Node &node,                  \
                                       const Inputs &inputs,              \
                                       const IOVars &inData,              \
                                       IOVars &outData, uint64_t ind) {   \
  vals[node] =                                                            \
      vals[inputs.at(node)[0].source] OP vals[inputs.at(node)[1].source]; \
  return true;                                                            \
}          

PROCESS_GENERIC_BITWISE_BINARY_OP_FUNC(And, &)

PROCESS_GENERIC_BITWISE_BINARY_OP_FUNC(Or, |)

PROCESS_GENERIC_BITWISE_BINARY_OP_FUNC(Xor, ^)

static bool processNotOp(RecordedValues &vals, const Node &node,
                         const Inputs &inputs, const IOVars &inData,
                         IOVars &outData, uint64_t ind) {
  vals[node] = ~(vals[inputs.at(node)[0].source]);
  return true;
}

static bool processNegOp(RecordedValues &vals, const Node &node,
                         const Inputs &inputs, const IOVars &inData,
                         IOVars &outData, uint64_t ind) {
  DFTypeImpl *type = (DFVariable(node.var).getType()).getImpl();
  if (type->isFixed()) {
    int64_t left =
        *(reinterpret_cast<int64_t*>(&vals[inputs.at(node)[0].source]));
    int64_t result = -left;
    vals[node] =
        *(reinterpret_cast<SimValue*>(&result));
  } else if (type->isFloat()) {
    double left =
        *(reinterpret_cast<double*>(&vals[inputs.at(node)[0].source]));
    double result = -left;
    vals[node] =
        *(reinterpret_cast<SimValue*>(&result));
  } else {
    return false;
  }
  return true;
}

static bool processShiftLeftOp(RecordedValues &vals, const Node &node,
                               const Inputs &inputs, const IOVars &inData,
                               IOVars &outData, uint64_t ind) {
  vals[node] = vals[inputs.at(node)[0].source] << node.data.bitShift;
  return true;
}

static bool processShiftRightOp(RecordedValues &vals, const Node &node,
                                const Inputs &inputs, const IOVars &inData,
                                IOVars &outData, uint64_t ind) {
  vals[node] = vals[inputs.at(node)[0].source] >> node.data.bitShift;
  return true;
}

bool DFCXXSimulator::processOp(RecordedValues &vals, const Node &node,
                               const IOVars &inData, IOVars &outData,
                               uint64_t ind) {
  if (funcs.find(node.type) == funcs.end()) {
    return false;
  }
  return funcs.at(node.type)(vals, node, inputs, inData, outData, ind);
}

DFCXXSimulator::DFCXXSimulator(std::vector<Node> &nodes,
                               const Inputs &inputs) :
                               nodes(nodes),
                               inputs(inputs),
                               funcs({
                                 {OpType::IN, processInput},
                                 {OpType::OUT, processOutput},
                                 {OpType::CONST, processConst},
                                 {OpType::MUX, processMux},
                                 {OpType::ADD, GENERIC_FUNC_NAME(Add)},
                                 {OpType::SUB, GENERIC_FUNC_NAME(Sub)},
                                 {OpType::MUL, GENERIC_FUNC_NAME(Mul)},
                                 {OpType::DIV, GENERIC_FUNC_NAME(Div)},
                                 {OpType::AND, GENERIC_FUNC_NAME(And)},
                                 {OpType::OR, GENERIC_FUNC_NAME(Or)},
                                 {OpType::XOR, GENERIC_FUNC_NAME(Xor)},
                                 {OpType::NOT, processNotOp},
                                 {OpType::NEG, processNegOp},
                                 {OpType::LESS, GENERIC_FUNC_NAME(Less)},
                                 {OpType::LESSEQ, GENERIC_FUNC_NAME(LessEq)},
                                 {OpType::GREATER, GENERIC_FUNC_NAME(Greater)},
                                 {
                                  OpType::GREATEREQ,
                                  GENERIC_FUNC_NAME(GreaterEq)},
                                 {OpType::EQ, GENERIC_FUNC_NAME(Eq)},
                                 {OpType::NEQ, GENERIC_FUNC_NAME(Neq)},
                                 {OpType::SHL, processShiftLeftOp},
                                 {OpType::SHR, processShiftRightOp}}) {
  // TODO: Add offset support in the future.
}

bool DFCXXSimulator::runSim(IOVars &inData,
                            IOVars &outData,
                            uint64_t count) {
  // Node->value mapping is initialized. This allows us
  // to rememeber the relevant value for the operand node.
  // With every single "clock" (loop iteration) input
  // nodes' mapping is updated with the value from the buffer.
  RecordedValues vals;
  for (uint64_t i = 0; i < count; ++i) {
    for (Node &node : nodes) {
      if (!processOp(vals, node, inData, outData, i)) {
        return false;
      }
    }
  }
  return true;
}

bool DFCXXSimulator::writeOutput(std::ofstream &out,
                                 const IOVars &outData,
                                 uint64_t count) {
  auto outFunc = [&out, &outData] (uint64_t iter) {
    for (const auto &kv : outData) {
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
  IOVars inData;
  IOVars outData;
  while (uint64_t count = readInput(in, inData)) {
    // If either the simulation itself or writing to output file
    // fails - return false.
    if (!runSim(inData, outData, count) ||
        !writeOutput(out, outData, count)) {
      return false;
    }
  }
  return true;
}

} // namespace dfcxx
