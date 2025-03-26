//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/simulator.h"
#include "dfcxx/types/types.h"
#include "dfcxx/vars/vars.h"

#include "ctemplate/template.h"

#include <ctime>
#include <sstream>
#include <string>

namespace dfcxx {

#define HEX_BASE 16

uint64_t DFCXXSimulator::readInput(std::ifstream &in,
                                   IOVars &inData) {
  uint64_t ind = 0;
  std::string line;
  bool atLeastOne = false;
  while (std::getline(in, line) && ind < SIM_DATA_BUF_SIZE) {
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
        std::stoul(line.substr(spaceInd + 1), 0, HEX_BASE);
  }
  // It is assumed that at least one data block exists.
  return atLeastOne ? (ind + 1) : 0;
}

static bool processInput(RecordedValues &vals, Node *node,
                         const IOVars &inData, uint64_t ind) {
  auto name = std::string(DFVariable(node->var).getName());
  vals[node] = inData.at(name)[ind];
  return true;
}

static bool processOutput(RecordedValues &vals, Node *node,
                          const IOVars &inData, uint64_t ind) {
  auto name = std::string(DFVariable(node->var).getName());
  // Take output's only connection and assign the existing source value.
  vals[node] = vals[node->inputs[0]->source];
  return true;
}

static bool processConst(RecordedValues &vals, Node *node,
                         const IOVars &inData, uint64_t ind) {
  vals[node] = ((DFConstant *) node->var)->getUInt();
  return true;
}

static bool processMux(RecordedValues &vals, Node *node,
                       const IOVars &inData, uint64_t ind) {
  auto muxedValue = vals[node->inputs[node->data.muxId]->source];
  vals[node] = vals[node->inputs[muxedValue + 1]->source];
  return true;
}

// Generic name for a simulation function.
#define GENERIC_FUNC_NAME(OP_NAME) process##OP_NAME##Name

// Casts the provided value to a concrete type in a system-dependent way. 
#define CAST_SIM_VALUE_TO(TYPE, VALUE) *(reinterpret_cast<TYPE *>(&VALUE))

// Generic procedure to perform the simulation
// for the concrete binary op and type of its operands.
#define GENERIC_BINARY_OP_SIM_WITH_TYPE(TYPE, VALS, NODE, OP)        \
TYPE left = CAST_SIM_VALUE_TO(TYPE, VALS[NODE->inputs[0]->source]);  \
TYPE right = CAST_SIM_VALUE_TO(TYPE, VALS[NODE->inputs[1]->source]); \
TYPE result = left OP right;                                         \
VALS[NODE] = CAST_SIM_VALUE_TO(SimValue, result);

// Generic procedure to perform the simulation
// for the concrete unary op and type of its operand.
#define GENERIC_UNARY_OP_SIM_WITH_TYPE(TYPE, VALS, NODE, OP)        \
TYPE left = CAST_SIM_VALUE_TO(TYPE, VALS[NODE->inputs[0]->source]); \
TYPE result = OP left;                                              \
VALS[NODE] = CAST_SIM_VALUE_TO(SimValue, result);

#define PROCESS_GENERIC_BINARY_OP_FUNC(OP_NAME, OP)               \
static bool GENERIC_FUNC_NAME(OP_NAME)(RecordedValues &vals,      \
                                       Node *node,                \
                                       const IOVars &inData,      \
                                       uint64_t ind) {            \
  DFTypeImpl *type = (DFVariable(node->var).getType()).getImpl(); \
  if (type->isFixed()) {                                          \
    if (((FixedType*) type)->isSigned()) {                        \
      GENERIC_BINARY_OP_SIM_WITH_TYPE(int64_t, vals, node, OP)    \
    } else {                                                      \
      GENERIC_BINARY_OP_SIM_WITH_TYPE(uint64_t, vals, node, OP)   \
    }                                                             \
  } else if (type->isFloat()) {                                   \
      GENERIC_BINARY_OP_SIM_WITH_TYPE(double, vals, node, OP)     \
  } else {                                                        \
    return false;                                                 \
  }                                                               \
  return true;                                                    \
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

#define PROCESS_GENERIC_BITWISE_BINARY_OP_FUNC(OP_NAME, OP)           \
static bool GENERIC_FUNC_NAME(OP_NAME)(RecordedValues &vals,          \
                                       Node *node,                    \
                                       const IOVars &inData,          \
                                       uint64_t ind) {                \
  vals[node] =                                                        \
      vals[node->inputs[0]->source] OP vals[node->inputs[1]->source]; \
  return true;                                                        \
}          

PROCESS_GENERIC_BITWISE_BINARY_OP_FUNC(And, &)

PROCESS_GENERIC_BITWISE_BINARY_OP_FUNC(Or, |)

PROCESS_GENERIC_BITWISE_BINARY_OP_FUNC(Xor, ^)

static bool processNotOp(RecordedValues &vals, Node *node,
                         const IOVars &inData, uint64_t ind) {
  vals[node] = ~(vals[node->inputs[0]->source]);
  return true;
}

static bool processNegOp(RecordedValues &vals, Node *node,
                         const IOVars &inData, uint64_t ind) {
  DFTypeImpl *type = (DFVariable(node->var).getType()).getImpl();
  if (type->isFixed()) {
    if (((FixedType*) type)->isSigned()) {
      GENERIC_UNARY_OP_SIM_WITH_TYPE(int64_t, vals, node, -)
    } else {
      GENERIC_UNARY_OP_SIM_WITH_TYPE(uint64_t, vals, node, -)
    }
  } else if (type->isFloat()) {
      GENERIC_UNARY_OP_SIM_WITH_TYPE(double, vals, node, -)
  } else {
    return false;
  }
  return true;
}

static bool processShiftLeftOp(RecordedValues &vals, Node *node,
                               const IOVars &inData, uint64_t ind) {
  vals[node] = vals[node->inputs[0]->source] << node->data.bitShift;
  return true;
}

static bool processShiftRightOp(RecordedValues &vals, Node *node,
                                const IOVars &inData, uint64_t ind) {
  vals[node] = vals[node->inputs[0]->source] >> node->data.bitShift;
  return true;
}

bool DFCXXSimulator::processOp(RecordedValues &vals, Node *node,
                               const IOVars &inData, uint64_t ind) {
  if (funcs.find(node->type) == funcs.end()) {
    return false;
  }
  return funcs.at(node->type)(vals, node, inData, ind);
}

DFCXXSimulator::DFCXXSimulator(std::vector<Node *> &nodes) :
                               nodes(nodes),
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

bool DFCXXSimulator::runSim(RecordedValues &vals,
                            IOVars &inData,
                            uint64_t iter) {
  // Node->value mapping is updated. This allows us
  // to remember the relevant value for the operand node.
  // With every single "clock" input nodes' mapping is updated
  // with the value from the buffer.
  for (Node *node : nodes) {
    if (!processOp(vals, node, inData, iter)) {
      return false;
    }
  }
  return true;
}

#define ASCII_ZERO_CHAR_NUM 48

static inline std::string valueToBinary(SimValue value, uint64_t width) {
  std::stringstream stream;
  uint64_t lastBitId = width - 1;
  // Every iteration we take leftmost bit and convert it to a char.
  for (uint64_t bitId = 0; bitId < width; ++bitId) {
    stream << char(((value >> (lastBitId - bitId)) & 1) + ASCII_ZERO_CHAR_NUM);
  }
  return stream.str();
} 

void
DFCXXSimulator::writeOutput(ctemplate::TemplateDictionary *dict,
                            const RecordedValues &vals,
                            uint64_t startInd,
                            uint64_t iter,
                            const std::unordered_map<Node *, std::string> &idMap) {
  ctemplate::TemplateDictionary *tick =
      dict->AddSectionDictionary("TICKS");
  tick->SetValue("TICK", std::to_string(startInd + iter));
  for (const auto &kv : vals) {
    ctemplate::TemplateDictionary *value =
        tick->AddSectionDictionary("VALUES");
    value->SetValue("VALUE",
                    valueToBinary(kv.second, 
                                  kv.first->var->getType()->getTotalBits()));
    value->SetValue("NAME", idMap.at(kv.first));
  }
}

void DFCXXSimulator::genHeader(ctemplate::TemplateDictionary *dict,
                               const RecordedValues &vals,
                               std::unordered_map<Node *, std::string> &idMap,
                               uint64_t &counter) {
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
  
  auto getName = [&idMap, &counter] (Node *node) -> std::string {
    // If the node is named - just save and return the name.
    auto name = DFVariable(node->var).getName();
    if (!name.empty()) {
      return (idMap[node] = name.data());
    }
    // If the mapping contains the node name - return it.
    auto it = idMap.find(node);
    if (it != idMap.end()) {
      return it->second;
    }
    // Otherwise create and return the new node name mapping. 
    return (idMap[node] = "node" + std::to_string(counter++));
  };
  
  for (const auto &kv : vals) {
    std::string name = getName(kv.first);
    auto width = kv.first->var->getType()->getTotalBits();
   
    ctemplate::TemplateDictionary *var =
        dict->AddSectionDictionary("VARS");
    var->SetValue("WIDTH",
                  std::to_string(width));
    var->SetValue("NAME", name);
  
    ctemplate::TemplateDictionary *initVar =
        dict->AddSectionDictionary("INIT_VARS");
    initVar->SetValue("INIT_VALUE", std::string(width, 'x'));
    initVar->SetValue("NAME", name);
  }
}

bool DFCXXSimulator::simulate(std::ifstream &in,
                              std::ofstream &out) {
  IOVars inData;
  RecordedValues vals;
  bool headerGenerated = false;
  uint64_t startInd = 1;
  uint64_t counter = 0;
  std::unordered_map<Node *, std::string> idMapping;
  ctemplate::TemplateDictionary *dict =
      new ctemplate::TemplateDictionary("vcd");

  while (uint64_t count = readInput(in, inData)) {
    for (uint64_t iter = 0; iter < count; ++iter) {
      // If the simulation fails - return false.
      if (!runSim(vals, inData, iter)) {
        delete dict;
        return false;
      }
      // If it's the first iteration - generate .vcd headers.
      if (!headerGenerated) {
        genHeader(dict, vals, idMapping, counter);
        headerGenerated = true;
      }
      writeOutput(dict, vals, startInd, iter, idMapping);
    }
    startInd += count;
  }
  dict->SetValue("FINAL_TICK", std::to_string(startInd));
  std::string result;
  ctemplate::ExpandTemplate(VCD_TEMPLATE_PATH,
                            ctemplate::DO_NOT_STRIP,
                            dict,
                            &result);

  out << result;

  delete dict;
  return true;
}

} // namespace dfcxx
