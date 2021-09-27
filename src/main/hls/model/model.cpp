/*
 * Copyright 2021 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#include <hls/model/model.h>

namespace eda::hls::model {

std::ostream& operator <<(std::ostream &out, const ChanType &chantype) {
  return out << chantype.datatype << "<" << chantype.maxflow << ">" << " " << chantype.name;
}

static std::ostream& operator <<(std::ostream &out, const std::vector<ChanType *> &args) {
  bool comma = false;
  for (const ChanType *chantype: args) {
    out << (comma ? ", " : "") << *chantype;
    comma = true;
  }

  return out;
}

std::ostream& operator <<(std::ostream &out, const NodeType &nodetype) {
  out << "nodetype<latency=" << nodetype.latency << "> " << nodetype.name;
  return out << "(" << nodetype.inputs << ") => (" << nodetype.outputs << ");";
}

std::ostream& operator <<(std::ostream &out, const Chan &chan) {
  return out << "chan " << chan.datatype << " " << chan.name << ";";
}

static std::ostream& operator <<(std::ostream &out, const std::vector<Chan *> &params) {
  bool comma = false;
  for (const Chan *chan: params) {
    out << (comma ? ", " : "") << chan->name;
    comma = true;
  }

  return out;
}

std::ostream& operator <<(std::ostream &out, const Node &node) {
  out << "node " << node.type.name;
  return out << "(" << node.inputs << ") => (" << node.outputs << ");";
}

std::ostream& operator <<(std::ostream &out, const Graph &graph) {
  out << "graph " << graph.name << " {" << std::endl;

  for (const Chan *chan: graph.chans) {
    out << "  " << *chan << std::endl;
  }

  for (const Node *node: graph.nodes) {
    out << "  " << *node << std::endl;
  }

  return out << "}";
}

std::ostream& operator <<(std::ostream &out, const Model &model) {
  for (const NodeType *nodetype: model.nodetypes) {
    out << *nodetype << std::endl;
  }

  for (const Graph *graph: model.graphs) {
    out << *graph << std::endl;
  }

  return out;
}

} // namespace eda::hls::model
