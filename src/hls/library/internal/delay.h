//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/element_internal.h"
#include "hls/library/library.h"

namespace eda::hls::library {

struct Delay final : public ElementInternal {
  static constexpr const char *width = "width";
  static constexpr const char *depth = "depth";

  Delay(const std::string &name,
        const Parameters &params,
        const std::vector<Port> &ports) :
  ElementInternal(name, params, ports) {}
  virtual ~Delay() = default;

  virtual void estimate(
      const Parameters &params, Indicators &indicators) const override;
  static std::shared_ptr<MetaElement> create(const NodeType &nodetype);
};

} // namespace eda::hls::library
