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

using namespace eda::hls::mapper;

namespace eda::hls::library {

struct Default final : public ElementInternal {
  static constexpr const char *stages = "stages";

  Default(const std::string &name,
          const Parameters &params,
          const std::vector<Port> &ports) :
  ElementInternal(name, params, ports) {}
  virtual ~Default() = default;

  virtual void estimate(
      const Parameters &params, Indicators &indicators) const override;
  virtual std::unique_ptr<Element> construct(
      const Parameters &params) const override;
  static std::shared_ptr<MetaElement> create(const NodeType &nodetype,
                                             const HWConfig &hwconfig);
};

} // namespace eda::hls::library
