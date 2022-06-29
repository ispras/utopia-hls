//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/parser/hil/parser.h"
#include "util/graph.h"

#include "gtest/gtest.h"

#include <iostream>
using namespace eda::utils::graph;
using namespace eda::hls::model;
using namespace eda::hls::parser::hil;

void traverseTest(const std::string &filename) {

  std::shared_ptr<Model> model = parse(filename);

  Graph *graph = model->main();

  ASSERT_TRUE(graph);

  std::cout << "Visiting nodes:" << std::endl;

  auto handleNode = [](const Node* node) {
    std::cout << node->name << std::endl;
  };

  auto handleEdge = [](const Chan* node) {};

  traverseTopologicalOrder<Graph, const Node*, const Chan*>(*graph, handleNode, handleEdge);
}

TEST(UtilTest, GraphTraverse) {
  traverseTest("test/data/hil/test.hil");
}

TEST(UtilTest, IdctTraverse) {
  traverseTest("test/data/hil/idct.hil");
}
