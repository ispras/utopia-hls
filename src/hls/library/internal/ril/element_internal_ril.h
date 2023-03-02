//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/element_internal.h"
#include "util/assert.h"

#include <string>
#include <vector>

namespace eda::hls::library::internal::ril {

using SharedMetaElement = std::shared_ptr<MetaElement>;

struct ElementInternalRil : public ElementInternal {

  ElementInternalRil(const std::string &name,
                     const std::string &libraryName,
                     const bool isCombinational,
                     const Parameters &params,
                     const std::vector<Port> &ports) :
  ElementInternal(name, libraryName, isCombinational, params, ports) {}

  virtual ~ElementInternalRil() = default;

  static SharedMetaElement create(const NodeType &nodetype,
                                  const HWConfig &hwconfig);

};

} // namespace eda::hls::library::internal::ril