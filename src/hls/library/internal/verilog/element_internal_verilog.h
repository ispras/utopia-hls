//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/element_internal.h"
#include "hls/library/library.h"
#include "util/assert.h"

#include <string>
#include <vector>

namespace eda::hls::library::internal::verilog {

struct ElementInternalVerilog : public ElementInternal {
  ElementInternalVerilog(const std::string &name,
                         const std::string &libraryName,
                         const bool isCombinational,
                         const Parameters &params,
                         const std::vector<Port> &ports) :
  ElementInternal(name, libraryName, isCombinational, params, ports) {}
  virtual ~ElementInternalVerilog() = default;
  static std::shared_ptr<MetaElement> create(const NodeType &nodetype,
                                             const HWConfig &hwconfig);
  static std::vector<std::shared_ptr<MetaElement>> createDefaultElements();
};

} // namespace eda::hls::library::internal::verilog
