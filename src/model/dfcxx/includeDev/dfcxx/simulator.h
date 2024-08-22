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

class DFCXXSimulator {
public:
  DFCXXSimulator(std::vector<Node> &nodes,
                 const Inputs &inputs);
  bool simulate(std::ifstream &in, std::ofstream &out);

private:
  uint64_t readInputData(std::ifstream &in, IOVars &inputMapping);
  bool runSim(IOVars &input, IOVars &output, uint64_t count);
  bool writeOutput(std::ofstream &out, IOVars &output, uint64_t count);

  void processInput(RecordedValues &vals, Node &node,
               IOVars &input, uint64_t ind);
  void processOutput(RecordedValues &vals, Node &node,
               IOVars &output, uint64_t ind);
  void processConst(RecordedValues &vals, Node &node);
  void processMux(RecordedValues &vals, Node &node);
  template <dfcxx::OpType T>
  void processBinaryOp(RecordedValues &vals, Node &node);
  template <dfcxx::OpType T>
  void processUnaryOp(RecordedValues &vals, Node &node);
  void processShiftLeft(RecordedValues &vals, Node &node);
  void processShiftRight(RecordedValues &vals, Node &node);

  std::vector<Node> &nodes;
  const Inputs &inputs;
  bool intermediateResults;
};

} // namespace dfcxx

#endif // DFCXX_SIMULATOR_H
