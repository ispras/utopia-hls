//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_SIMULATOR_H
#define DFCXX_SIMULATOR_H

#include "dfcxx/graph.h"
#include "dfcxx/node.h"

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

// This forward declaration is needed to avoid
// users having to include CTemplate headers.
namespace ctemplate {
  class TemplateDictionary;
}

namespace dfcxx {

typedef uint64_t SimValue;

// Buffer size for reading from and writing to simulation data files. 
#define SIM_DATA_BUF_SIZE 200

typedef std::map<std::string, std::array<SimValue, SIM_DATA_BUF_SIZE>> IOVars;

typedef std::unordered_map<Node *, std::vector<Channel *>> Inputs;

typedef std::unordered_map<Node *, SimValue> RecordedValues;

typedef bool (*OpSimulationFunc)(RecordedValues &vals, Node *node,
                                 const Inputs &inputs, const IOVars &inData,
                                 uint64_t ind);

typedef std::unordered_map<OpType, OpSimulationFunc> OpSimulationFuncs;

class DFCXXSimulator {
public:
  DFCXXSimulator(std::vector<Node *> &nodes,
                 const Inputs &inputs);
  bool simulate(std::ifstream &in, std::ofstream &out);

private:
  uint64_t readInput(std::ifstream &in, IOVars &inData);
  bool runSim(RecordedValues &vals, IOVars &inData, uint64_t iter);
  
  bool processOp(RecordedValues &vals, Node *node,
                 const IOVars &inData, uint64_t ind);
  
  void genHeader(ctemplate::TemplateDictionary *dict,
                 const RecordedValues &vals,
                 std::unordered_map<Node *, std::string> &idMap,
                 uint64_t &counter);

  void writeOutput(ctemplate::TemplateDictionary *dict,
                   const RecordedValues &vals,
                   uint64_t startInd,
                   uint64_t iter,
                   const std::unordered_map<Node *, std::string> &idMap);

  std::vector<Node *> &nodes;
  const Inputs &inputs;
  const OpSimulationFuncs funcs;
};

} // namespace dfcxx

#endif // DFCXX_SIMULATOR_H
