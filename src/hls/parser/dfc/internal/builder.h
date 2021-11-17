//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/parser/dfc/stream.h"
#include "util/singleton.h"

class Graph;
class Model;

namespace eda::hls::parser::dfc {

class Builder final: public eda::util::Singleton<Builder> {
public:
  void startKernel(const std::string &name);
  void endKernel();

  void addStream(const shared_ptr<dfc::stream> &stream);

private:
  Model model;
};

} // eda::hls::parser::dfc
