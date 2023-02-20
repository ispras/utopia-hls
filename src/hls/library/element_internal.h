//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "util/assert.h"

#include <string>
#include <vector>

namespace eda::hls::library::internal {

struct ElementInternal : public MetaElement {
  ElementInternal(const std::string &name,
                  const std::string &libraryName,
                  const bool isCombinational,
                  const Parameters &params,
                  const std::vector<Port> &ports) :
  MetaElement(name, libraryName, isCombinational, params, ports) {}
  virtual ~ElementInternal() = default;
  protected:
    static std::vector<Port> createPorts(const NodeType &nodetype);
};
} // namespace eda::hls::library::internal
