//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

namespace mdl = eda::hls::model;
class TestHilModel final {
public:

  static mdl::Model* get() {

    mdl::Model *model = new mdl::Model("test");
    mdl::Graph *graph = new mdl::Graph("main", *model);

    model->addGraph(graph);

    // Create node types.
    mdl::NodeType *source  = new mdl::NodeType("source",  *model);
    mdl::NodeType *split   = new mdl::NodeType("split",   *model);
    mdl::NodeType *kernel1 = new mdl::NodeType("kernel1", *model);
    mdl::NodeType *kernel2 = new mdl::NodeType("kernel2", *model);
    mdl::NodeType *merge   = new mdl::NodeType("merge",   *model);
    mdl::NodeType *sink    = new mdl::NodeType("sink",    *model);

    // Create ports.
    mdl::Port *srcX = new mdl::Port("x", Type::get("X"), 1.0, 0, false, 0);
    mdl::Port *srcY = new mdl::Port("y", Type::get("Y"), 1.0, 0, false, 0);
    source->addOutput(srcX);
    source->addOutput(srcY);

    mdl::Port   *splInX = new mdl::Port("x",  mdl::Type::get("X"), 1.0, 0, false, 0);
    mdl::Port *splOutX1 = new mdl::Port("x1", mdl::Type::get("X"), 0.5, 1, false, 0);
    mdl::Port *splOutX2 = new mdl::Port("x2", mdl::Type::get("X"), 0.5, 1, false, 0);
    split->addInput(splInX);
    split->addOutput(splOutX1);
    split->addOutput(splOutX2);

    mdl::Port  *kern1InX = new mdl::Port("x", mdl::Type::get("X"), 1.0,  0, false, 0);
    mdl::Port  *kern1InY = new mdl::Port("y", mdl::Type::get("Y"), 0.5,  0, false, 0);
    mdl::Port *kern1OutZ = new mdl::Port("z", mdl::Type::get("Z"), 0.25, 1, false, 0);
    mdl::Port *kern1OutW = new mdl::Port("w", mdl::Type::get("W"), 1.0,  2, false, 0);
    kernel1->addInput(kern1InX);
    kernel1->addInput(kern1InY);
    kernel1->addOutput(kern1OutZ);
    kernel1->addOutput(kern1OutW);

    mdl::Port  *kern2InX = new mdl::Port("x", mdl::Type::get("X"), 0.5,  0, false, 0);
    mdl::Port  *kern2InW = new mdl::Port("w", mdl::Type::get("W"), 0.5,  0, false, 0);
    mdl::Port *kern2OutZ = new mdl::Port("z", mdl::Type::get("Z"), 0.25, 1, false, 0);
    kernel2->addInput(kern2InX);
    kernel2->addInput(kern2InW);
    kernel2->addOutput(kern2OutZ);
    
    mdl::Port *mrgInZ1 = new mdl::Port("z1", mdl::Type::get("Z"), 0.5, 0, false, 0);
    mdl::Port *mrgInZ2 = new mdl::Port("z2", mdl::Type::get("Z"), 0.5, 0, false, 0);
    mdl::Port *mrgOutZ = new mdl::Port("z",  mdl::Type::get("Z"), 1.0, 1, false, 0);
    merge->addInput(mrgInZ1);
    merge->addInput(mrgInZ2);
    merge->addOutput(mrgOutZ);

    mdl::Port *sinkZ = new mdl::Port("z", mdl::Type::get("Z"), 1.0, 1, false, 0);
    sink->addInput(sinkZ);

    mdl::Signature sourceSignature = source->getSignature();
    mdl::Signature splitSignature = split->getSignature();
    mdl::Signature kernel1Signature = kernel1->getSignature();
    mdl::Signature kernel2Signature = kernel2->getSignature();
    mdl::Signature mergeSignature = merge->getSignature();
    mdl::Signature sinkSignature = sink->getSignature();

    model->addNodetype(sourceSignature, source);
    model->addNodetype(splitSignature, split);
    model->addNodetype(kernel1Signature, kernel1);
    model->addNodetype(kernel2Signature, kernel2);
    model->addNodetype(mergeSignature, merge);
    model->addNodetype(sinkSignature, sink);

    // Create channels.
    mdl::Chan *x1 = new mdl::Chan("x1", "X", *graph);
    mdl::Chan *x2 = new mdl::Chan("x2", "X", *graph);
    mdl::Chan  *x = new mdl::Chan( "x", "X", *graph);
    mdl::Chan  *y = new mdl::Chan( "y", "Y", *graph);
    mdl::Chan *z1 = new mdl::Chan("z1", "Z", *graph);
    mdl::Chan *z2 = new mdl::Chan("z2", "Z", *graph);
    mdl::Chan  *z = new mdl::Chan( "z", "Z", *graph);
    mdl::Chan  *w = new mdl::Chan( "w", "W", *graph);

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
    mdl::Node *n1 = new mdl::Node("n1", *source, *graph);
    mdl::Node *n2 = new mdl::Node("n2", *split, *graph);
    mdl::Node *n3 = new mdl::Node("n3", *kernel1, *graph);
    mdl::Node *n4 = new mdl::Node("n4", *kernel2, *graph);
    mdl::Node *n5 = new mdl::Node("n5", *merge, *graph);
    mdl::Node *n6 = new mdl::Node("n6", *sink, *graph);

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
