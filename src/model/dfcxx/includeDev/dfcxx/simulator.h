//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
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

namespace dfcxx {

typedef uint64_t SimValue;

// Buffer size for reading from and writing to simulation data files. 
#define BUF_SIZE 200

typedef std::map<std::string, std::array<SimValue, BUF_SIZE>> IOVars;

typedef std::unordered_map<Node, std::vector<Channel>> Inputs;

typedef std::unordered_map<Node, SimValue> RecordedValues;

typedef bool (*OpSimulationFunc)(RecordedValues &vals, const Node &node,
                                 const Inputs &inputs, const IOVars &inData,
                                 IOVars &outData, uint64_t ind);

typedef std::unordered_map<OpType, OpSimulationFunc> OpSimulationFuncs;

class DFCXXSimulator {
public:
  DFCXXSimulator(std::vector<Node> &nodes,
                 const Inputs &inputs);
  bool simulate(std::ifstream &in, std::ofstream &out);

private:
  uint64_t readInput(std::ifstream &in, IOVars &inData);
  bool runSim(IOVars &inData, IOVars &outData, uint64_t count);
  
  bool processOp(RecordedValues &vals, const Node &node,
                 const IOVars &inData, IOVars &outData, uint64_t ind);
  
  bool writeOutput(std::ofstream &out, const IOVars &outData, uint64_t count);

  std::vector<Node> &nodes;
  const Inputs &inputs;
  const OpSimulationFuncs funcs;
};

} // namespace dfcxx

#endif // DFCXX_SIMULATOR_H
