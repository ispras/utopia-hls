//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <hls/model/model.h>
#include <string>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

class LpSolver final {

public:

  void setModel(const std::string& fileName);

  void balance();

private:

  void checkFlows(const Node* node);
  
  Model* model;

};

const std::string MERGE = "merge";
const std::string SPLIT = "split";
int isMergeSplit(const std::string& nodeName);
bool checkMerge(const std::string &nodeName);
bool checkSplit(const std::string &nodeName);


} // namespace eda::hls::scheduler
