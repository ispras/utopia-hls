//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "HIL/API.h"
#include "HIL/Dumper.h"
#include "hls/model/model.h"

#include <gtest/gtest.h>

using eda::hls::model::Binding;
using eda::hls::model::Chan;
using eda::hls::model::Graph;
using eda::hls::model::Model;
using eda::hls::model::Node;
using eda::hls::model::NodeType;
using eda::hls::model::Port;
using eda::hls::model::Signature;
template<typename Type>
using Transformer = mlir::transforms::Transformer<Type>;

class TransformTest : public ::testing::Test {
protected:
  void SetUp() override {
    /*               n2[fst]              */
    /* n1[source]┌───>┌─┐                 */
    /*         ┌─┤x[X]└─┘┐z[Z]            */
    /*         └─┤       └───>┌─┐         */
    /*           └───>┌─┐────>└─┘n4[sink] */
    /*            y[Y]└─┘ w[W]            */
    /*               n3[snd]              */

    // Create model.
    model = std::make_unique<Model>("M");
    // Create graph.
    auto graph = new Graph{"main", *model};
    // Create chans.
    auto chan_x = new Chan{"x", "X", *graph};
    auto chan_y = new Chan{"y", "Y", *graph};
    auto chan_z = new Chan{"z", "Z", *graph};
    auto chan_w = new Chan{"w", "W", *graph};
    // Create ports.
    auto port_x = Port{"x", chan_x->type, 1.0f, 0, false, 0};
    auto port_y = Port{"y", chan_y->type, 1.0f, 0, false, 0};
    auto port_z = Port{"z", chan_z->type, 1.0f, 0, false, 0};
    auto port_w = Port{"w", chan_w->type, 1.0f, 0, false, 0};
    // Create nodetypes.
    auto source_nt = new NodeType{"source", *model};
    source_nt->addOutput(new Port{port_x});
    source_nt->addOutput(new Port{port_y});
    Signature source_nt_signature = source_nt->getSignature();
    auto fst_nt = new NodeType{"fst", *model};
    fst_nt->addInput(new Port{port_x});
    fst_nt->addOutput(new Port{port_z});
    Signature fst_nt_signature = fst_nt->getSignature();
    auto snd_nt = new NodeType{"snd", *model};
    snd_nt->addInput(new Port{port_y});
    snd_nt->addOutput(new Port{port_w});
    Signature snd_nt_signature = snd_nt->getSignature();
    auto sink_nt = new NodeType("sink", *model);
    sink_nt->addInput(new Port{port_z});
    sink_nt->addInput(new Port{port_w});
    Signature sink_nt_signature = sink_nt->getSignature();
    // Add nodetypes.
    model->addNodetype(source_nt_signature, source_nt);
    model->addNodetype(fst_nt_signature, fst_nt);
    model->addNodetype(snd_nt_signature, snd_nt);
    model->addNodetype(sink_nt_signature, sink_nt);
    // Create nodes.
    auto n1 = new Node{"n1", *source_nt, *graph};
    auto n2 = new Node{"n2", *fst_nt, *graph};
    auto n3 = new Node{"n3", *snd_nt, *graph};
    auto n4 = new Node{"n4", *sink_nt, *graph};
    // Tie chans to nodes.
    n1->addOutput(chan_x);
    n1->addOutput(chan_y);
    n2->addInput(chan_x);
    n2->addOutput(chan_z);
    n3->addInput(chan_y);
    n3->addOutput(chan_w);
    n4->addInput(chan_z);
    n4->addInput(chan_w);
    // Tie chans to node ports.
    chan_x->source = {n1, new Port{port_x}};
    chan_x->target = {n2, new Port{port_x}};
    chan_y->source = {n1, new Port{port_y}};
    chan_y->target = {n3, new Port{port_y}};
    chan_z->source = {n2, new Port{port_z}};
    chan_z->target = {n4, new Port{port_z}};
    chan_w->source = {n3, new Port{port_w}};
    chan_w->target = {n4, new Port{port_w}};
    // Add nodes to graph.
    graph->addNode(n1);
    graph->addNode(n2);
    graph->addNode(n3);
    graph->addNode(n4);
    // Add chans to graph.
    graph->addChan(chan_x);
    graph->addChan(chan_y);
    graph->addChan(chan_z);
    graph->addChan(chan_w);
    // Add graph to model.
    model->addGraph(graph);
  }

  void TearDown() override {
    if (!model) {
      return;
    }
    // Free memory.
    for (auto graph : model->graphs) {
      for (auto node : graph->nodes) {
        delete node;
      }
      for (auto chan : graph->chans) {
        delete chan;
      }
      delete graph;
    }
    for (auto nodeTypeIterator = model->nodetypes.begin();
         nodeTypeIterator != model->nodetypes.end();
         nodeTypeIterator++) {
      const auto *nodetype = nodeTypeIterator->second;
      for (auto input_port : nodetype->inputs) {
        delete input_port;
      }
      for (auto output_port : nodetype->outputs) {
        delete output_port;
      }
      delete nodetype;
    }
  }

  std::unique_ptr<Model> model;
};

TEST_F(TransformTest, CheckName) { EXPECT_EQ(model->name, "M"); }

TEST_F(TransformTest, InsertDelay) {
  std::cout << *model << std::endl;
  Transformer<Model> transformer{*model};
  transformer.applyTransform(mlir::transforms::ChanAddSourceTarget());
  transformer.applyTransform(mlir::transforms::InsertDelay("x", 7));
  auto modelAfter = transformer.done();
  std::cout << modelAfter << std::endl;
  EXPECT_EQ(modelAfter.name, "M");
}

TEST_F(TransformTest, InsertDelayUndo) {
  std::cout << *model << std::endl;
  Transformer<Model> transformer{*model};
  transformer.applyTransform(mlir::transforms::ChanAddSourceTarget());
  transformer.applyTransform(mlir::transforms::InsertDelay("x", 7));
  transformer.undoTransforms();
  auto modelAfter = transformer.done();
  std::cout << modelAfter << std::endl;
  EXPECT_EQ(modelAfter.name, "M");
}