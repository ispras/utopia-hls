//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "util/assert.h"

#include <string>
#include <vector>

using namespace eda::hls::mapper;

namespace eda::hls::library {

struct ElementInternal : public MetaElement {
  ElementInternal(const std::string &name,
                  const Parameters &params,
                  const std::vector<Port> &ports) :
  MetaElement(name, params, ports) {}
  virtual ~ElementInternal() = default;
  static std::shared_ptr<MetaElement> create(const NodeType &nodetype,
                                             const HWConfig &hwconfig);
  protected:
    static std::vector<Port> createPorts(const NodeType &nodetype);
};

} // namespace eda::hls::library
