//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

using namespace eda::hls::model;

class TestHilModel final {
public:

  static Model* get() {

    Model *model = new Model("test");
    Graph *graph = new Graph("main", *model);

    model->addGraph(graph);

    // Create node types.
    NodeType *source  = new NodeType("source",  *model);
    NodeType *split   = new NodeType("split",   *model);
    NodeType *kernel1 = new NodeType("kernel1", *model);
    NodeType *kernel2 = new NodeType("kernel2", *model);
    NodeType *merge   = new NodeType("merge",   *model);
    NodeType *sink    = new NodeType("sink",    *model);

    model->addNodetype(source);
    model->addNodetype(split);
    model->addNodetype(kernel1);
    model->addNodetype(kernel2);
    model->addNodetype(merge);
    model->addNodetype(sink);

    // Create ports.
    eda::hls::model::Port *srcX = new eda::hls::model::Port("x", Type::get("X"), 1.0, 0, false, 0);
    eda::hls::model::Port *srcY = new eda::hls::model::Port("y", Type::get("Y"), 1.0, 0, false, 0);
    source->addOutput(srcX);
    source->addOutput(srcY);

    eda::hls::model::Port   *splInX = new eda::hls::model::Port("x",  Type::get("X"), 1.0, 0, false, 0);
    eda::hls::model::Port *splOutX1 = new eda::hls::model::Port("x1", Type::get("X"), 0.5, 1, false, 0);
    eda::hls::model::Port *splOutX2 = new eda::hls::model::Port("x2", Type::get("X"), 0.5, 1, false, 0);
    split->addInput(splInX);
    split->addOutput(splOutX1);
    split->addOutput(splOutX2);

    eda::hls::model::Port  *kern1InX = new eda::hls::model::Port("x", Type::get("X"), 1.0,  0, false, 0);
    eda::hls::model::Port  *kern1InY = new eda::hls::model::Port("y", Type::get("Y"), 0.5,  0, false, 0);
    eda::hls::model::Port *kern1OutZ = new eda::hls::model::Port("z", Type::get("Z"), 0.25, 1, false, 0);
    eda::hls::model::Port *kern1OutW = new eda::hls::model::Port("w", Type::get("W"), 1.0,  2, false, 0);
    kernel1->addInput(kern1InX);
    kernel1->addInput(kern1InY);
    kernel1->addOutput(kern1OutZ);
    kernel1->addOutput(kern1OutW);

    eda::hls::model::Port  *kern2InX = new eda::hls::model::Port("x", Type::get("X"), 0.5,  0, false, 0);
    eda::hls::model::Port  *kern2InW = new eda::hls::model::Port("w", Type::get("W"), 0.5,  0, false, 0);
    eda::hls::model::Port *kern2OutZ = new eda::hls::model::Port("z", Type::get("Z"), 0.25, 1, false, 0);
    kernel2->addInput(kern2InX);
    kernel2->addInput(kern2InW);
    kernel2->addOutput(kern2OutZ);
    
    eda::hls::model::Port *mrgInZ1 = new eda::hls::model::Port("z1", Type::get("Z"), 0.5, 0, false, 0);
    eda::hls::model::Port *mrgInZ2 = new eda::hls::model::Port("z2", Type::get("Z"), 0.5, 0, false, 0);
    eda::hls::model::Port *mrgOutZ = new eda::hls::model::Port("z",  Type::get("Z"), 1.0, 1, false, 0);
    merge->addInput(mrgInZ1);
    merge->addInput(mrgInZ2);
    merge->addOutput(mrgOutZ);

    eda::hls::model::Port *sinkZ = new eda::hls::model::Port("z", Type::get("Z"), 1.0, 1, false, 0);
    sink->addInput(sinkZ);

    // Create channels.
    Chan *x1 = new Chan("x1", "X", *graph);
    Chan *x2 = new Chan("x2", "X", *graph);
    Chan  *x = new Chan( "x", "X", *graph);
    Chan  *y = new Chan( "y", "Y", *graph);
    Chan *z1 = new Chan("z1", "Z", *graph);
    Chan *z2 = new Chan("z2", "Z", *graph);
    Chan  *z = new Chan( "z", "Z", *graph);
    Chan  *w = new Chan( "w", "W", *graph);

    x->ind  = {0, 0};
    y->ind  = {0, 0};
    x1->ind = {1, 0};
    x2->ind = {1, 0};
    z1->ind = {1, 0};
    z2->ind = {1, 0};
    z->ind  = {1, 0};
    w->ind  = {2, 0};

    graph->addChan(x);
    graph->addChan(x1);
    graph->addChan(x2);
    graph->addChan(y);
    graph->addChan(z);
    graph->addChan(z1);
    graph->addChan(z2);
    graph->addChan(w);

    // Create nodes.
    Node *n1 = new Node("n1", *source, *graph);
    Node *n2 = new Node("n2", *split, *graph);
    Node *n3 = new Node("n3", *kernel1, *graph);
    Node *n4 = new Node("n4", *kernel2, *graph);
    Node *n5 = new Node("n5", *merge, *graph);
    Node *n6 = new Node("n6", *sink, *graph);

    n1->addOutput(x);
    n1->addOutput(y);

    n2->addInput(x);
    n2->addOutput(x1);
    n2->addOutput(x2);

    n3->addInput(x1);
    n3->addInput(y);
    n3->addOutput(z1);
    n3->addOutput(w);

    n4->addInput(x2);
    n4->addInput(w);
    n4->addOutput(z2);

    n5->addInput(z1);
    n5->addInput(z2);
    n5->addOutput(z);

    n6->addInput(z);

    x->source  = {n1, srcX};
    x->target  = {n2, splInX};

    x1->source = {n2, splOutX1};
    x1->target = {n3, kern1InX};

    x2->source = {n2, splOutX2};
    x2->target = {n4, kern2InX};

    y->source  = {n1, srcY};
    y->target  = {n3, kern1InY};

    z1->source = {n3, kern1OutZ};
    z1->target = {n5, mrgInZ1};

    z2->source = {n4, kern2OutZ};
    z2->target = {n5, mrgInZ2};

    z->source  = {n5, mrgOutZ};
    z->target  = {n6, sinkZ};

    w->source  = {n3, kern1OutW};
    w->target  = {n4, kern2InW};

    graph->addNode(n1);
    graph->addNode(n2);
    graph->addNode(n3);
    graph->addNode(n4);
    graph->addNode(n5);
    graph->addNode(n6);
    return model;
  }

private:
  TestHilModel() {};

};
