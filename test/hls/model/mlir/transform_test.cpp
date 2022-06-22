//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
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

class SimpleModel : public ::testing::Test {
protected:
  void SetUp() override {
    /*               n2[fst]              */
    /* n1[source]┌───>┌─┐                 */
    /*         ┌─┤x[X]└─┘┐z[Z]            */
    /*         └─┤       └───>┌─┐         */
    /*           └───>┌─┐────>└─┘n4[sink] */
    /*            y[Y]└─┘ w[W]            */
    /*               n3[snd]              */

    // create model
    model_ = std::make_unique<Model>("M");
    // create graph
    auto graph = new Graph{"main", *model_};
    // create chans
    auto chan_x = new Chan{"x", "X", *graph};
    auto chan_y = new Chan{"y", "Y", *graph};
    auto chan_z = new Chan{"z", "Z", *graph};
    auto chan_w = new Chan{"w", "W", *graph};
    // create ports
    auto port_x = Port{"x", chan_x->type, 1.0f, 0, false, 0};
    auto port_y = Port{"y", chan_y->type, 1.0f, 0, false, 0};
    auto port_z = Port{"z", chan_z->type, 1.0f, 0, false, 0};
    auto port_w = Port{"w", chan_w->type, 1.0f, 0, false, 0};
    // create nodetypes
    auto source_nt = new NodeType{"source", *model_};
    source_nt->addOutput(new Port{port_x});
    source_nt->addOutput(new Port{port_y});
    auto fst_nt = new NodeType{"fst", *model_};
    fst_nt->addInput(new Port{port_x});
    fst_nt->addOutput(new Port{port_z});
    auto snd_nt = new NodeType{"snd", *model_};
    snd_nt->addInput(new Port{port_y});
    snd_nt->addOutput(new Port{port_w});
    auto sink_nt = new NodeType("sink", *model_);
    sink_nt->addInput(new Port{port_z});
    sink_nt->addInput(new Port{port_w});
    // add nodetypes
    model_->addNodetype(source_nt);
    model_->addNodetype(fst_nt);
    model_->addNodetype(snd_nt);
    model_->addNodetype(sink_nt);
    // create nodes
    auto n1 = new Node{"n1", *source_nt, *graph};
    auto n2 = new Node{"n2", *fst_nt, *graph};
    auto n3 = new Node{"n3", *snd_nt, *graph};
    auto n4 = new Node{"n4", *sink_nt, *graph};
    // tie chans to nodes
    n1->addOutput(chan_x);
    n1->addOutput(chan_y);
    n2->addInput(chan_x);
    n2->addOutput(chan_z);
    n3->addInput(chan_y);
    n3->addOutput(chan_w);
    n4->addInput(chan_z);
    n4->addInput(chan_w);
    // tie chans to node ports
    chan_x->source = {n1, &port_x};
    chan_x->target = {n2, &port_x};
    chan_y->source = {n1, &port_y};
    chan_y->target = {n3, &port_y};
    chan_z->source = {n2, &port_z};
    chan_z->target = {n4, &port_z};
    chan_w->source = {n3, &port_w};
    chan_w->target = {n4, &port_w};
    // add nodes to graph
    graph->addNode(n1);
    graph->addNode(n2);
    graph->addNode(n3);
    graph->addNode(n4);
    // add chans to graph
    graph->addChan(chan_x);
    graph->addChan(chan_y);
    graph->addChan(chan_z);
    graph->addChan(chan_w);
    // add graph to model
    model_->addGraph(graph);
  }

  void TearDown() override {
    if (!model_) {
      return;
    }
    // Free memory
    for (auto graph : model_->graphs) {
      for (auto node : graph->nodes) {
        delete node;
      }
      for (auto chan : graph->chans) {
        delete chan;
      }
      delete graph;
    }
    for (auto nodetype : model_->nodetypes) {
      for (auto input_port : nodetype->inputs) {
        delete input_port;
      }
      for (auto output_port : nodetype->outputs) {
        delete output_port;
      }
      delete nodetype;
    }
  }

  std::unique_ptr<Model> model_;
};

TEST_F(SimpleModel, CheckName) { EXPECT_EQ(model_->name, "M"); }

TEST_F(SimpleModel, InsertDelay) {
  using namespace mlir::transforms;
  Transformer<Model> transformer{*model_};
  transformer.apply_transform(ChanAddSourceTarget());
  transformer.apply_transform(InsertDelay("x", 7));
  auto model_after = transformer.done();
  std::cout << model_after << std::endl;
}

TEST_F(SimpleModel, InsertDelayUndo) {
  using namespace mlir::transforms;
  Transformer<Model> transformer{*model_};
  transformer.apply_transform(ChanAddSourceTarget());
  transformer.apply_transform(InsertDelay("x", 7));
  transformer.undo_transforms();
  auto model_after = transformer.done();
  EXPECT_EQ(model_after.name, "M");
}
