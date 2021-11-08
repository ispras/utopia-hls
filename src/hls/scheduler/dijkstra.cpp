//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/dijkstra.h"

#include <iostream>

namespace eda::hls::scheduler {

DijkstraBalancer::~DijkstraBalancer() {
  deleteEntries();
}

void DijkstraBalancer::deleteEntries() {
  /*for (; !pathElements.empty(); pathElements.pop()) {
    delete pathElements.top();
  }

  for (auto* node : ready) {
    delete node;
  }*/

  for (auto entry : nodeMap) {
    delete entry.second;
  }
}

void DijkstraBalancer::reset() {

  deleteEntries();  

  // Reset the queues
  /*pathElements = std::priority_queue<PathNode*, std::vector<PathNode*>, 
      PathNode::PathNodeCmp>();
  ready = std::vector<PathNode*>();*/
  nodeMap = std::map<const Node*, PathNode*>();
}

void addNeighbours(std::vector<std::pair<Chan*, unsigned>> &dest, 
    const std::vector<Chan*> &neighbours) {
  for (auto *chan : neighbours) {
    unsigned latency = chan->source.port->latency;
    dest.push_back(std::make_pair(chan, latency));
  }
}

void DijkstraBalancer::init(const Graph* graph) {
  reset();
  // Init the elements
  for (const auto *node : graph->nodes) {
    PathNode *pNode = new PathNode(0/*PathNode::getInitialValue(*node)*/);
    //pathElements.push(pNode);
    nodeMap[node] = pNode;
    
    addNeighbours(pNode->successors, node->outputs);
    addNeighbours(pNode->predessors, node->inputs);
  }
}



/*void DijkstraBalancer::relax(const PathNode *src,
    std::pair<const Node*, unsigned> &dst) {
  unsigned curTime = src->nodeTime;
  PathNode *dstNode = nodeMap[dst.first];
  unsigned dstTime = curTime + dst.second;

  std::cout << "curTime: " << src->nodeTime;
  std::cout << " curTime+latency: " << dstTime;
  std::cout << " dst node: " << dst.first->name;
  std::cout << " dst node time: " << dstNode->nodeTime <<"\n";

  if (dstTime < dstNode->nodeTime) {
    dstNode->nodeTime = dstTime;
    std::cout << "set dst time: " << dstNode->nodeTime<<"\n";
  }
}*/

/*std::ostream& operator <<(std::ostream &out, const std::map<const Node*, PathNode*> &nodes) {
  
  for (const auto node : nodes) {
    const Node *orig = node.first;
    const PathNode *pNode = node.second;

    out << "Src node: " << orig->name;
    out << " timestep: " << pNode->nodeTime << "\n";
    out << "Successors:\n";
    for (const auto &succ : pNode->successors) {
      out << "  node: " << succ.first->name << " latency: " << succ.second << "\n";
    }


  }
  return out;
}*/

void DijkstraBalancer::visit(PathNode *node) {

  toVisit.insert(toVisit.end(), node->successors.begin(), node->successors.end());

  for (const auto &next : node->successors) {
    unsigned curTime = node->nodeTime;
    PathNode *dstNode = nodeMap[next.first->target.node];
    unsigned dstTime = curTime + next.second;

    /*std::cout << "curTime: " << curTime;
    std::cout << " curTime+latency: " << dstTime;
    std::cout << " dst node: " << next.first->name;
    std::cout << " dst node time: " << dstNode->nodeTime << "\n";*/


    if (dstTime > dstNode->nodeTime) {
      dstNode->nodeTime = dstTime;
      //std::cout << "set dst time: " << dstNode->nodeTime << "\n";
    }
  }
}

void DijkstraBalancer::balance(Model &model) {
  for (const auto *graph : model.graphs) {
    if (graph->isMain()) {
      init(graph);

      PathNode* nextNode = nullptr;
      for (const auto *node : graph->nodes) {
        if (node->isSource()) {
          nextNode = nodeMap[node];
          break;
        }
      }

      if (nextNode != nullptr) {
        do {
          visit(nextNode);
          nextNode = nodeMap[toVisit.front().first->target.node];
          toVisit.pop_front();
        } while (!toVisit.empty());
      }
    }
  }
  /*std::cout << nodeMap << "\n";
  std::cout << "end\n";*/
  insertBuffers(model);
}

void DijkstraBalancer::insertBuffers(Model &model) {

  for (const auto &node : nodeMap) {
    const unsigned curTime = node.second->nodeTime;
    for (const auto &pred : node.second->predessors) {
      const PathNode *predNode = nodeMap[pred.first->source.node];
      unsigned delta = curTime - (predNode->nodeTime + pred.second);
      if (delta > 0) {
        model.insertDelay(*pred.first, delta);
      }
    }
  }

}


/*void DijkstraBalancer::balance(Model &model) {
  for (const auto *graph : model.graphs) {
    if (graph->isMain()) {
      init(graph);
      std::cout << nodeMap << "\n";
      for (; !pathElements.empty(); pathElements.pop()) {
        PathNode *node = pathElements.top();
        for (auto succ : node->successors) {
          //relax(node, succ);
        }
        ready.push_back(node);
      }
      std::cout << nodeMap << "\n";
      std::cout << "end\n";
    }
  }
}*/



} // namespace eda::hls::scheduler
