//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>

namespace eda::hls::model {

struct Chan;
struct Model;
struct Node;
struct NodeType;
struct Port;

struct Transform {
  Transform(Model &model): model(model) {}
  virtual ~Transform() {}

  virtual void apply() = 0;
  virtual void undo() = 0;

  Model &model;
};

struct InsertDelay: public Transform {
  InsertDelay(Model &model, Chan &chan, unsigned latency):
    Transform(model), chan(chan), latency(latency) {}

  virtual void apply() override;
  virtual void undo() override;
  
  Chan &chan;
  unsigned latency;

  std::vector<NodeType*> newTypes;
  std::vector<Port*> newPorts;
  std::vector<Chan*> newChans;
  std::vector<Node*> newNodes;
};

} // namespace eda::hls::model
